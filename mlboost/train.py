from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure we can import from root
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.db import DbConfig, FlightRepository
from mlboost.features import FlightAggregator
from mlboost.model import XGBoostAnomalyModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost Flight Anomaly Detector")
    parser.add_argument("--db", type=Path, default=Path("last.db"), help="Main flight DB")
    parser.add_argument("--table", type=str, default="flight_tracks", help="Normal flights table")
    parser.add_argument("--anomalies-table", type=str, default="anomalous_tracks", help="Anomalous flights table")
    parser.add_argument("--model-path", type=Path, default=Path("mlboost/output/xgb_model.joblib"))
    parser.add_argument("--limit", type=int, default=None, help="Limit flights per class for speed")
    return parser.parse_args()

def load_dataset(repo, label, limit=None):
    print(f"Loading from {repo._config.table} (limit={limit})...")
    flights = list(repo.iter_flights(limit=limit, min_points=10))
    flights = [f for f in flights if f.points] # Ensure not empty
    print(f"Loaded {len(flights)} flights.")
    
    aggregator = FlightAggregator()
    df = aggregator.process_flights(flights, label=label)
    return df

def run_training(db_path: Path, output_path: Path, limit=None):
    args = argparse.Namespace(
        db=db_path,
        table="flight_tracks",
        anomalies_table="anomalous_tracks",
        model_path=output_path,
        limit=limit
    )
    main(args)

def main(args=None):
    if args is None:
        args = parse_args()
    
    # 1. Load Data
    print("=== Data Loading ===")
    # Ensure DB path is correct relative to current run location or absolute
    if not args.db.exists():
        print(f"Warning: {args.db} not found. Trying absolute path or relative to root...")
        # Fallback logic if needed, but assuming run from root
    
    repo_normal = FlightRepository(DbConfig(path=args.db, table=args.table))
    repo_anom = FlightRepository(DbConfig(path=args.db, table=args.anomalies_table))
    
    # Load and aggregate
    df_normal = load_dataset(repo_normal, label=0, limit=args.limit)
    df_anom = load_dataset(repo_anom, label=1, limit=args.limit)
    
    if df_normal.empty or df_anom.empty:
        print("Error: One of the datasets is empty. Check DB paths and table names.")
        return

    # Combine
    df_full = pd.concat([df_normal, df_anom], ignore_index=True)
    
    # Shuffle
    df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total Dataset: {len(df_full)} rows (Normal={len(df_normal)}, Anomaly={len(df_anom)})")
    
    # 2. Prepare for Training
    # Drop metadata that isn't a feature
    drop_cols = ["flight_id", "label"]
    feature_cols = [c for c in df_full.columns if c not in drop_cols]
    
    X = df_full[feature_cols]
    y = df_full["label"]
    
    # Train/Test Split (Stratified to keep ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Further split train into train / calibration sets for probability calibration
    X_train_inner, X_cal, y_train_inner, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=43,
    )
    
    print(f"\nTrain Size (inner): {len(X_train_inner)}")
    print(f"Calib Size:         {len(X_cal)}")
    print(f"Test Size:          {len(X_test)}")
    
    # 3. Train Model
    print("\n=== Training ===")
    # Ensure output directory exists
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model = XGBoostAnomalyModel(model_path=args.model_path)
    model.fit(X_train_inner, y_train_inner)

    # 3b. Calibrate probabilities on a held-out calibration set
    # Use isotonic by default; fall back gracefully if calib set is single-class
    model.calibrate(X_cal, y_cal, method="isotonic")
    
    # 4. Evaluate
    print("\n=== Evaluation (Test Set) ===")
    model.evaluate(X_test, y_test)
    
    # 5. Save (This replaces the old model)
    model.save()
    print(f"Model saved to {args.model_path}")
    
    # 6. Feature Importance (if available)
    print("\n=== Top 10 Features ===")
    base_estimator = getattr(model.model, "base_estimator", model.model)
    if hasattr(base_estimator, "feature_importances_"):
        importance = base_estimator.feature_importances_
        feats = model.feature_names
        sorted_idx = np.argsort(importance)[::-1]
        for i in range(min(10, len(feats))):
            idx = sorted_idx[i]
            print(f"{i+1}. {feats[idx]}: {importance[idx]:.4f}")
    else:
        print("Feature importances not available for the calibrated model; skipping.")

if __name__ == "__main__":
    main()

