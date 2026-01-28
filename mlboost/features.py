from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from .point_features import FeatureExtractor

class FlightAggregator:
    """
    Aggregates point-level features into flight-level statistics
    suitable for XGBoost classification.
    """
    
    def __init__(self):
        self.point_extractor = FeatureExtractor()
        
    def extract_flight_row(self, flight, label: int = None) -> Dict[str, Any]:
        """
        Process a single flight into a flat dictionary of features.
        """
        # Get point-level features
        points_data = self.point_extractor.extract_flight_features(flight)
        if not points_data:
            return {}
            
        df = pd.DataFrame(points_data)
        
        # Define columns to aggregate
        # We skip metadata like 'flight_id', 'timestamp' for aggregation
        # We skip One-Hot encoded phases for general stats, but maybe sum them?
        numeric_cols = [
            "alt", "gspeed", "vspeed", "climb_rate", 
            "speed_accel", "vert_accel", "turn_rate", "vspeed_abs",
            "cum_turn_300", "cum_alt_60", "avg_speed_300"
        ]
        
        row = {}
        if label is not None:
            row["label"] = label
            
        row["flight_id"] = flight.flight_id
        row["duration"] = df["timestamp"].max() - df["timestamp"].min()
        row["num_points"] = len(df)
        
        # Calculate stats for each numeric column
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if series.empty:
                continue
                
            row[f"{col}_mean"] = series.mean()
            row[f"{col}_std"] = series.std(ddof=0) # Population std usually fine
            row[f"{col}_min"] = series.min()
            row[f"{col}_max"] = series.max()
            row[f"{col}_q95"] = series.quantile(0.95)
            row[f"{col}_q05"] = series.quantile(0.05)
            
        # Phase percentages (how much time spent in each phase?)
        # The point extractor returns phase_ground, phase_climb, etc.
        phase_cols = ["phase_ground", "phase_climb", "phase_descent", "phase_cruise"]
        for p_col in phase_cols:
            if p_col in df.columns:
                row[f"{p_col}_pct"] = df[p_col].mean()
                
        return row

    def process_flights(self, flights, label: int = None) -> pd.DataFrame:
        """
        Process a list of flights into a DataFrame.
        """
        rows = []
        for f in flights:
            try:
                row = self.extract_flight_row(f, label=label)
                if row:
                    rows.append(row)
            except Exception as e:
                print(f"Error processing flight {f.flight_id}: {e}")
                continue
                
        return pd.DataFrame(rows)

