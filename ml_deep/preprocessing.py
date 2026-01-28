from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from core.models import FlightTrack, TrackPoint

class TrajectoryResampler:
    """
    Resamples a flight trajectory to a fixed number of points (K).
    This creates a consistent vector size for clustering and neural networks.
    """
    
    def __init__(self, num_points: int = 50):
        self.num_points = num_points
        
    def process(self, flight: FlightTrack) -> pd.DataFrame:
        """
        Resamples the flight to exactly `num_points` equidistant in time.
        Returns a DataFrame with shape (num_points, features).
        """
        points = flight.sorted_points()
        if not points or len(points) < 2:
            return pd.DataFrame()
            
        # Extract raw arrays
        timestamps = np.array([p.timestamp for p in points])
        
        # Remove duplicates in timestamps to avoid divide-by-zero in interpolation
        # and ensure strict monotonicity if possible
        unique_indices = np.unique(timestamps, return_index=True)[1]
        # np.unique sorts the unique values, so if timestamps was sorted, this is fine.
        # But let's ensure we keep the order corresponding to sorted time.
        unique_indices.sort()
        
        if len(unique_indices) < 2:
             return pd.DataFrame()
             
        points = [points[i] for i in unique_indices]
        timestamps = timestamps[unique_indices]

        # Normalize time to 0..1
        t_min, t_max = timestamps[0], timestamps[-1]
        if t_max == t_min:
            return pd.DataFrame()
            
        t_norm = (timestamps - t_min) / (t_max - t_min)
        
        # Target timestamps (0.0, 0.02, ..., 1.0)
        t_target = np.linspace(0, 1, self.num_points)
        
        # Features to interpolate
        data = {
            "lat": np.array([p.lat for p in points]),
            "lon": np.array([p.lon for p in points]),
            "alt": np.array([p.alt or 0.0 for p in points]),
            "gspeed": np.array([p.gspeed or 0.0 for p in points]),
            "track": np.array([p.track or 0.0 for p in points]),
        }
        
        resampled = {"flight_id": flight.flight_id}
        
        # Interpolate each feature
        for name, values in data.items():
            # Replace None/NaN with 0.0 before interpolation to be safe
            values = np.nan_to_num(values, nan=0.0)
            
            # Handle cyclical nature of heading/track
            if name == "track":
                # Unwind angles to avoid 359->1 jump issues
                values = np.unwrap(np.radians(values))
                f = interp1d(t_norm, values, kind='linear', fill_value="extrapolate")
                interpolated = f(t_target)
                # Convert back to degrees 0-360
                resampled[name] = np.degrees(interpolated) % 360
            else:
                f = interp1d(t_norm, values, kind='linear', fill_value="extrapolate")
                resampled[name] = f(t_target)
            
            # Check for NaNs/Infs after interpolation
            if np.isnan(resampled[name]).any() or np.isinf(resampled[name]).any():
                print(f"WARNING: NaN/Inf detected in {name} after interpolation!")
                resampled[name] = np.nan_to_num(resampled[name], nan=0.0, posinf=0.0, neginf=0.0)
                
        # Add relative time
        resampled["progress"] = t_target
        
        return pd.DataFrame(resampled)

    def flatten(self, df: pd.DataFrame) -> np.ndarray:
        """
        Flattens the trajectory into a 1D vector [lat1, lon1, alt1... latK, lonK, altK]
        Useful for Clustering.
        """
        # Select features for clustering (geometry only usually)
        # We use Lat, Lon, Alt, Track
        features = ["lat", "lon", "alt", "track"]
        matrix = df[features].values # Shape (K, 4)
        return matrix.flatten() # Shape (K*4,)

    def to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Returns (Seq_Len, Features)"""
        features = ["lat", "lon", "alt", "track"]
        return df[features].values

