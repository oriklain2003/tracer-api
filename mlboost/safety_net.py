from __future__ import annotations

import argparse
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Ensure we can import from root
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.db import DbConfig, FlightRepository
from mlboost.features import FlightAggregator

def parse_args():
    parser = argparse.ArgumentParser(description="Train Unsupervised Safety Net (Isolation Forest)")
    parser.add_argument("--db", type=Path, default=Path("last.db"))
    parser.add_argument("--table", type=str, default="flight_tracks")
    parser.add_argument("--anomalies-table", type=str, default="anomalous_tracks")
    parser.add_argument("--contamination", type=float, default=0.01, help="Expected rate of anomalies in normal data")
    return parser.parse_args()

def load_data(repo, label, limit=None):
    print(f"Loading from {repo._config.table}...")
    flights = list(repo.iter_flights(limit=limit, min_points=10))
    aggregator = FlightAggregator()
    # We don't pass label effectively since IF is unsupervised, but useful for eval
    df = aggregator.process_flights(flights, label=label)
    return df

def main():
    args = parse_args()
    
    # 1. Load Data
    repo_normal = FlightRepository(DbConfig(path=args.db, table=args.table))
    repo_anom = FlightRepository(DbConfig(path=args.db, table=args.anomalies_table))
    
    # Load Normal Data (Training Set)
    print("--- Loading Normal Data ---")
    df_normal = load_data(repo_normal, label=0, limit=None)
    
    # Load Anomaly Data (Test Set ONLY)
    print("--- Loading Anomaly Data ---")
    df_anom = load_data(repo_anom, label=1, limit=None)
    
    # 2. Prepare Training Data (Only Normal!)
    # We assume the "normal" database is mostly clean (e.g., 99% clean)
    features = [c for c in df_normal.columns if c not in ["flight_id", "label"]]
    
    X_train = df_normal[features]
    
    # 3. Train Isolation Forest
    print(f"\nTraining Isolation Forest on {len(X_train)} normal flights...")
    # contamination determines the threshold. 0.01 means we assume 1% of "normal" data is actually noise.
    iso_forest = IsolationForest(
        n_estimators=200, 
        contamination=args.contamination, 
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    # 4. Evaluate
    print("\n=== Testing on Known Anomalies ===")
    # We test on the Known Anomalies + a subset of Normal data held out? 
    # Actually, let's just see how many of the Anomalies it catches.
    
    X_anom = df_anom[features]
    
    # Predict (-1 = Anomaly, 1 = Normal)
    preds_anom = iso_forest.predict(X_anom)
    
    # Convert to 0 (Normal) and 1 (Anomaly)
    y_pred_anom = [1 if x == -1 else 0 for x in preds_anom]
    
    recall = sum(y_pred_anom) / len(y_pred_anom)
    print(f"Recall on Known Anomalies: {recall:.2%} ({sum(y_pred_anom)}/{len(y_pred_anom)})")
    
    print("\n=== False Alarm Check (on Training Data) ===")
    # Ideally we do this on a holdout set, but let's see how "clean" it thinks the training set is
    preds_normal = iso_forest.predict(X_train)
    y_pred_normal = [1 if x == -1 else 0 for x in preds_normal]
    fpr = sum(y_pred_normal) / len(y_pred_normal)
    print(f"Flag Rate on Normal Data:  {fpr:.2%} ({sum(y_pred_normal)}/{len(y_pred_normal)})")
    print(f"(Target was ~{args.contamination:.2%})")
    
    # 5. Feature Contribution (Heuristic)
    # Isolation Forest doesn't give feature importance directly, 
    # but we can look at the flights it flagged and see which features are extreme.
    
    print("\n=== Sample Detected Anomaly Analysis ===")
    # Let's take the first anomaly caught and see why
    caught_indices = [i for i, x in enumerate(y_pred_anom) if x == 1]
    if caught_indices:
        idx = caught_indices[0]
        flight_row = df_anom.iloc[idx]
        print(f"Flight ID: {flight_row['flight_id']}")
        
        # Compare to Normal Means
        means = df_normal[features].mean()
        stds = df_normal[features].std()
        
        print("Features with > 2 Sigma deviation:")
        for col in features:
            val = flight_row[col]
            z_score = (val - means[col]) / (stds[col] + 1e-6)
            if abs(z_score) > 2:
                print(f"  - {col}: {val:.2f} (Z={z_score:.1f})")
                
    # Save
    model_path = Path("mlboost/output/safety_net.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso_forest, model_path)
    print(f"\nSafety Net saved to {model_path}")

if __name__ == "__main__":
    main()

