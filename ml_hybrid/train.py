import sys
import argparse
import logging
import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler

# Add root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.db import DbConfig, FlightRepository
from ml_deep.preprocessing import TrajectoryResampler
from ml_hybrid.model import HybridAutoencoder


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger("hybrid_train")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level)
    log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to: {log_file}")
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Anomaly Detector")
    parser.add_argument("--db", type=Path, default=Path("last.db"), help="Path to last.db")
    parser.add_argument("--table", type=str, default="flight_tracks", help="Table name to read from")
    parser.add_argument("--output-dir", type=Path, default=Path("ml_hybrid/output"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of flights")
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log = setup_logging(args.output_dir)
    
    log.info("=" * 60)
    log.info("Hybrid Autoencoder Training Started")
    log.info("=" * 60)
    log.info(f"Configuration:")
    log.info(f"  Database: {args.db}")
    log.info(f"  Table: {args.table}")
    log.info(f"  Epochs: {args.epochs}")
    log.info(f"  Limit: {args.limit or 'None (all flights)'}")
    log.info(f"  Output Dir: {args.output_dir}")
    
    # 1. Load Data
    log.info("-" * 40)
    log.info("Step 1: Loading Flight Data...")
    if not args.db.exists():
        log.error(f"Database not found: {args.db}")
        return

    repo = FlightRepository(DbConfig(path=args.db, table=args.table))
    flights = list(repo.iter_flights(limit=args.limit, min_points=50))
    
    if not flights:
        log.error("No flights found in database.")
        return
        
    log.info(f"Loaded {len(flights)} flights from database.")
    log.debug(f"Flight IDs sample: {[f.flight_id for f in flights[:5]]}")

    # 2. Preprocess
    log.info("-" * 40)
    log.info("Step 2: Resampling and Extracting Features...")
    resampler = TrajectoryResampler(num_points=50)
    
    vectors = []
    valid_count = 0
    skipped_count = 0
    
    # Features: Lat, Lon, Alt, GSpeed
    feature_cols = ["lat", "lon", "alt", "gspeed"]
    log.info(f"Feature columns: {feature_cols}")
    
    for i, f in enumerate(flights):
        df = resampler.process(f)
        if not df.empty:
            # Extract (50, 4)
            mat = df[feature_cols].values
            vectors.append(mat)
            valid_count += 1
        else:
            skipped_count += 1
            log.debug(f"Skipped flight {f.flight_id}: empty after resampling")
        
        if (i + 1) % 500 == 0:
            log.info(f"  Processed {i + 1}/{len(flights)} flights...")
            
    log.info(f"Valid flights: {valid_count}, Skipped: {skipped_count}")
            
    if not vectors:
        log.error("No valid vectors generated after preprocessing.")
        return
        
    # Shape: (N, 50, 4)
    X = np.array(vectors)
    log.info(f"Data Shape: {X.shape} (samples, sequence_len, features)")
    log.debug(f"Data stats - Min: {X.min():.4f}, Max: {X.max():.4f}, Mean: {X.mean():.4f}")
    
    # 3. Scale Data
    log.info("-" * 40)
    log.info("Step 3: Scaling Data...")
    # We need to scale features. Since it's 3D, we reshape to 2D, scale, then reshape back.
    N, Seq, Feat = X.shape
    X_flat = X.reshape(N * Seq, Feat)
    
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, Seq, Feat)
    
    log.debug(f"Scaler means: {scaler.mean_}")
    log.debug(f"Scaler scales: {scaler.scale_}")
    
    # Save Scaler
    scaler_path = args.output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved to: {scaler_path}")
    
    # 4. Train
    log.info("-" * 40)
    log.info("Step 4: Training Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    dataset = TensorDataset(torch.FloatTensor(X_scaled))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    log.info(f"DataLoader: {len(loader)} batches, batch_size=32")
    
    model = HybridAutoencoder(input_dim=Feat, seq_len=Seq).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        total_loss = 0
        batch_losses = []
        
        for batch in loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            
            recon = model(x_batch)
            # MSE Loss
            loss = torch.mean((x_batch - recon) ** 2)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
        avg_loss = total_loss / len(loader)
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            improvement = "★"
        else:
            improvement = ""
        
        log.info(f"Epoch {epoch+1:3d}/{args.epochs}: Loss={avg_loss:.6f} {improvement}")
        log.debug(f"  Batch loss range: [{min(batch_losses):.6f}, {max(batch_losses):.6f}]")
        
    log.info(f"Training complete. Best loss: {best_loss:.6f}")
        
    # 5. Determine Threshold (process in batches to avoid OOM)
    log.info("-" * 40)
    log.info("Step 5: Computing Anomaly Threshold...")
    model.eval()
    all_errors = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.FloatTensor(X_scaled[i:i+batch_size]).to(device)
            errors = model.get_reconstruction_error(batch).cpu().numpy()
            all_errors.extend(errors)
            
    all_errors = np.array(all_errors)
    
    log.debug(f"Reconstruction error stats:")
    log.debug(f"  Min: {all_errors.min():.6f}")
    log.debug(f"  Max: {all_errors.max():.6f}")
    log.debug(f"  Mean: {all_errors.mean():.6f}")
    log.debug(f"  Std: {all_errors.std():.6f}")
    log.debug(f"  Percentiles: 50th={np.percentile(all_errors, 50):.6f}, 95th={np.percentile(all_errors, 95):.6f}, 99th={np.percentile(all_errors, 99):.6f}")
        
    # Set threshold at 99th percentile
    threshold = float(np.quantile(all_errors, 0.99))
    log.info(f"Anomaly Threshold (99th percentile): {threshold:.6f}")
    
    # Save Artifacts
    log.info("-" * 40)
    log.info("Step 6: Saving Artifacts...")
    
    model_path = args.output_dir / "hybrid_model.pth"
    threshold_path = args.output_dir / "threshold.joblib"
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(threshold, threshold_path)
    
    log.info(f"  Model: {model_path}")
    log.info(f"  Threshold: {threshold_path}")
    
    # Save training metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "database": str(args.db),
        "table": args.table,
        "epochs": args.epochs,
        "samples": N,
        "sequence_length": Seq,
        "features": feature_cols,
        "threshold": threshold,
        "best_loss": best_loss,
        "device": str(device),
        "model_params": total_params
    }
    metadata_path = args.output_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Metadata: {metadata_path}")
    
    log.info("=" * 60)
    log.info("Training Complete!")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
