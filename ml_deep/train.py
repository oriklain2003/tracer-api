from __future__ import annotations

import argparse
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.db import DbConfig, FlightRepository
from ml_deep.preprocessing import TrajectoryResampler
from ml_deep.clustering import TrajectoryClusterer
from ml_deep.model import TrajectoryAutoencoder

def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep Autoencoder Anomaly Detector")
    parser.add_argument("--db", type=Path, default=Path("last.db"), help="Path to last.db")
    parser.add_argument("--output-dir", type=Path, default=Path("ml_deep/output"), help="Output directory for models")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of flights")
    return parser.parse_args()

def run_training(db_path: Path, output_dir: Path, epochs=50, limit=None):
    args = argparse.Namespace(
        db=db_path,
        output_dir=output_dir,
        clusters=5,
        epochs=epochs,
        limit=limit
    )
    main(args)

def main(args=None):
    if args is None:
        args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading Normal Flights from last.db...")
    if not args.db.exists():
        print(f"Warning: {args.db} not found. Please check the path.")
    
    repo = FlightRepository(DbConfig(path=args.db, table="flight_tracks"))
    flights = list(repo.iter_flights(limit=args.limit, min_points=50))
    
    if not flights:
        print("No flights found.")
        return
        
    # 2. Preprocess & Resample
    print(f"Resampling {len(flights)} flights...")
    resampler = TrajectoryResampler(num_points=50)
    
    vectors = []
    valid_flights = []
    
    for f in flights:
        df = resampler.process(f)
        if not df.empty:
            vec = resampler.flatten(df)
            vectors.append(vec)
            valid_flights.append(f.flight_id)
            
    X = np.array(vectors) # Shape (N, 200) if 50 points * 4 features
    print(f"Feature Matrix Shape: {X.shape}")
    
    # 3. Clustering (Flow Detection)
    cluster_model_path = args.output_dir / "clusters.joblib"
    clusterer = TrajectoryClusterer(n_clusters=args.clusters)
    clusterer.fit(X)
    labels = clusterer.predict(X)
    clusterer.save(cluster_model_path)
    print(f"Saved Cluster Model to {cluster_model_path}")
    
    # 4. Train Autoencoder per Cluster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    thresholds = {}
    
    for cid in range(args.clusters):
        # Get flights in this cluster
        mask = labels == cid
        X_cluster = X[mask]
        
        if len(X_cluster) < 10:
            print(f"Skipping Cluster {cid}: Too few samples ({len(X_cluster)})")
            continue
            
        print(f"\n=== Cluster {cid} (N={len(X_cluster)}) ===")
        
        # Prepare Tensors
        # Note: We reuse the clusterer's scaler to normalize for the AE too!
        # It's cleaner to have 0-mean inputs for Neural Nets
        X_norm = clusterer.scaler.transform(X_cluster)
        dataset = TensorDataset(torch.FloatTensor(X_norm))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Init Model
        input_dim = X.shape[1]
        model = TrajectoryAutoencoder(input_dim=input_dim).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        
        # Train Loop
        for epoch in range(args.epochs):
            total_loss = 0
            for batch in loader:
                x_batch = batch[0].to(device)
                optimizer.zero_grad()
                
                recon = model(x_batch)
                loss = torch.mean((x_batch - recon) ** 2)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.6f}")
                
        # Determine Anomaly Threshold for this Cluster
        # Use a high quantile of the reconstruction error distribution instead of
        # assuming it is Gaussian (mean + k * std). This is more robust and
        # directly controls the expected false alarm rate on normal data.
        model.eval()
        with torch.no_grad():
            x_all = torch.FloatTensor(X_norm).to(device)
            errors = model.get_reconstruction_error(x_all).cpu().numpy()
            
        # e.g. 99.5th percentile → ~0.5% of normal flights would exceed threshold
        quantile = 0.995
        threshold = float(np.quantile(errors, quantile))

        thresholds[str(cid)] = float(threshold)
        print(f"Threshold (q={quantile:.3f}): {threshold:.6f}")
        
        # Save Model
        torch.save(model.state_dict(), args.output_dir / f"ae_cluster_{cid}.pt")
        
    # Save Thresholds
    with open(args.output_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
