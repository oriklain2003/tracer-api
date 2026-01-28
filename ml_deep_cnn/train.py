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
from ml_deep_cnn.preprocessing import TrajectoryResampler
from ml_deep_cnn.clustering import TrajectoryClusterer
from ml_deep_cnn.model import TrajectoryCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep CNN Anomaly Detector")
    parser.add_argument("--db", type=Path, default=Path("last.db"), help="Path to last.db")
    parser.add_argument("--output-dir", type=Path, default=Path("ml_deep_cnn/output"), help="Output directory for models")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs") # More epochs for deep model
    parser.add_argument("--limit", type=int, default=None, help="Limit number of flights")
    return parser.parse_args()

def run_training(db_path: Path, output_dir: Path, epochs=100, limit=None):
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
        
    # 2. Preprocess
    print(f"Resampling {len(flights)} flights...")
    resampler = TrajectoryResampler(num_points=50)
    
    vectors_flat = [] # For clustering
    matrices = []     # For CNN [50, 4]
    
    for f in flights:
        df = resampler.process(f)
        if not df.empty:
            # Flat for clustering
            vectors_flat.append(resampler.flatten(df))
            # Matrix for CNN
            matrices.append(resampler.to_matrix(df))
            
    X_flat = np.array(vectors_flat)
    X_tensor = np.array(matrices) # [N, 50, 4]
    
    # 3. Clustering (on flat vectors)
    print("Clustering flows...")
    clusterer = TrajectoryClusterer(n_clusters=args.clusters)
    clusterer.fit(X_flat)
    labels = clusterer.predict(X_flat)
    clusterer.save(args.output_dir / "clusters.joblib")
    
    # 4. Train CNN per Cluster
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    thresholds = {}
    
    for cid in range(args.clusters):
        mask = labels == cid
        X_cluster = X_tensor[mask] # [N_c, 50, 4]
        
        if len(X_cluster) < 10:
            print(f"Skipping Cluster {cid}: Too few samples ({len(X_cluster)})")
            continue
            
        print(f"\n=== Cluster {cid} (N={len(X_cluster)}) ===")
        
        # Normalize: Use simple Global Stats or Per-Cluster Stats?
        # Let's use Per-Cluster Standardization for the CNN input
        # Shape [N, 50, 4] -> Normalize along (0, 1) is risky.
        # Better: Normalize each feature across the whole cluster.
        # Mean/Std shape: [1, 1, 4]
        
        mean = np.mean(X_cluster, axis=(0, 1), keepdims=True)
        std = np.std(X_cluster, axis=(0, 1), keepdims=True) + 1e-6
        
        # Save normalization stats for this cluster
        norm_stats = {"mean": mean.tolist(), "std": std.tolist()}
        with open(args.output_dir / f"norm_{cid}.json", "w") as f:
            json.dump(norm_stats, f)
            
        X_norm = (X_cluster - mean) / std
        
        dataset = TensorDataset(torch.FloatTensor(X_norm))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = TrajectoryCNN(num_features=4, seq_len=50).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        
        # Train
        for epoch in range(args.epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                loss = torch.mean(model.get_reconstruction_error(x))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        # Threshold
        model.eval()
        with torch.no_grad():
            x_all = torch.FloatTensor(X_norm).to(device)
            errors = model.get_reconstruction_error(x_all).cpu().numpy()
            
        # Use a high quantile of the error distribution for a robust cutoff
        quantile = 0.995
        threshold = float(np.quantile(errors, quantile))
        thresholds[str(cid)] = threshold
        print(f"Threshold (q={quantile:.3f}): {threshold:.6f}")
        
        torch.save(model.state_dict(), args.output_dir / f"cnn_{cid}.pt")
        
    with open(args.output_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f)

    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
