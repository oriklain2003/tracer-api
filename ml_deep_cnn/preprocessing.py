from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from core.models import FlightTrack

class TrajectoryResampler:
    """
    Resamples a flight trajectory to a fixed number of points (K).
    Same logic as before, just copied to the new folder to keep it standalone.
    """
    
    def __init__(self, num_points: int = 50):
        self.num_points = num_points
        
    def process(self, flight: FlightTrack) -> pd.DataFrame:
        points = flight.sorted_points()
        if not points or len(points) < 2:
            return pd.DataFrame()
            
        timestamps = np.array([p.timestamp for p in points])
        
        # Remove duplicates
        unique_indices = np.unique(timestamps, return_index=True)[1]
        unique_indices.sort()
        
        if len(unique_indices) < 2:
            return pd.DataFrame()
            
        points = [points[i] for i in unique_indices]
        timestamps = timestamps[unique_indices]

        t_min, t_max = timestamps[0], timestamps[-1]
        if t_max == t_min:
            return pd.DataFrame()
            
        t_norm = (timestamps - t_min) / (t_max - t_min)
        t_target = np.linspace(0, 1, self.num_points)
        
        data = {
            "lat": np.array([p.lat for p in points]),
            "lon": np.array([p.lon for p in points]),
            "alt": np.array([p.alt or 0.0 for p in points]),
            "gspeed": np.array([p.gspeed or 0.0 for p in points]),
            "track": np.array([p.track or 0.0 for p in points]),
        }
        
        resampled = {"flight_id": flight.flight_id}
        
        for name, values in data.items():
            # Replace None/NaN with 0.0 before interpolation to be safe
            values = np.nan_to_num(values, nan=0.0)

            if name == "track":
                values = np.unwrap(np.radians(values))
                f = interp1d(t_norm, values, kind='linear', fill_value="extrapolate")
                interpolated = f(t_target)
                resampled[name] = np.degrees(interpolated) % 360
            else:
                f = interp1d(t_norm, values, kind='linear', fill_value="extrapolate")
                resampled[name] = f(t_target)

            if np.isnan(resampled[name]).any() or np.isinf(resampled[name]).any():
                print(f"WARNING: NaN/Inf detected in {name} after interpolation!")
                resampled[name] = np.nan_to_num(resampled[name], nan=0.0, posinf=0.0, neginf=0.0)
                
        resampled["progress"] = t_target
        return pd.DataFrame(resampled)

    def flatten(self, df: pd.DataFrame) -> np.ndarray:
        # We use 4 features for the neural net
        features = ["lat", "lon", "alt", "track"]
        matrix = df[features].values
        return matrix.flatten()

    def to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Returns (Seq_Len, Features)"""
        features = ["lat", "lon", "alt", "track"]
        return df[features].values

