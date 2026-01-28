from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from .features import FlightAggregator
from .model import XGBoostAnomalyModel

class XGBoostDetector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
            
        self.model = XGBoostAnomalyModel.load(model_path)
        self.aggregator = FlightAggregator()
        
    def predict(self, flight) -> Dict[str, Any]:
        """
        Predict anomaly score for a single flight.
        """
        # 1. Extract Features
        row = self.aggregator.extract_flight_row(flight)
        if not row:
            return {"error": "Could not extract features (empty flight?)"}
            
        # 2. Prepare DataFrame
        df = pd.DataFrame([row])
        
        # 3. Align Columns (Fill missing with 0)
        for c in self.model.feature_names:
            if c not in df.columns:
                df[c] = 0
        df = df[self.model.feature_names]
        
        # 4. Predict
        prob = self.model.predict_proba(df)[0]
        is_anom = prob > 0.5
        
        return {
            "score": float(prob),
            "threshold": 0.5,
            "is_anomaly": bool(is_anom),
            "status": "ANOMALY" if is_anom else "NORMAL"
        }

