import torch
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union

from core.models import FlightTrack
from ml_hybrid.model import HybridAutoencoder

class HybridAnomalyDetector:
    def __init__(self, model_dir: Path):
        self.model_path = model_dir / "hybrid_model.pth"
        self.scaler_path = model_dir / "scaler.joblib"
        self.threshold_path = model_dir / "threshold.joblib"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.threshold = 0.05 # Default
        self.num_points = 50
        self.num_features = 4
        
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Hybrid model not found at {self.model_path}")
            
        # Load Scaler
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            
        # Load Threshold
        if self.threshold_path.exists():
            self.threshold = joblib.load(self.threshold_path)
            
        # Load Model
        # We need to know input dim from saved args or assume 4 (lat, lon, alt, speed)
        self.model = HybridAutoencoder(input_dim=self.num_features, seq_len=self.num_points)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, flight: FlightTrack) -> torch.Tensor:
        # Extract features: Lat, Lon, Alt, GSpeed
        # Normalize using scaler
        # Pad/Truncate to seq_len=50
        
        points = flight.sorted_points()
        data = []
        for p in points:
            data.append([p.lat, p.lon, p.alt, p.gspeed or 0])
            
        arr = np.array(data)
        
        # Scale
        if self.scaler:
            arr = self.scaler.transform(arr)
            
        # Fixed size 50
        target_len = self.num_points
        current_len = len(arr)
        
        if current_len > target_len:
            # Take middle or sample? Let's take middle for anomaly context or just first 50?
            # Usually we want the whole track. Let's interpolate.
            # For simplicity in this demo, we take the first 50 or pad.
            # Better: Resample.
            indices = np.linspace(0, current_len - 1, target_len).astype(int)
            arr = arr[indices]
        elif current_len < target_len:
            # Pad with last value
            padding = np.tile(arr[-1], (target_len - current_len, 1))
            arr = np.vstack([arr, padding])
            
        # Convert to Tensor [1, Seq, Feat]
        tensor = torch.FloatTensor(arr).unsqueeze(0).to(self.device)
        return tensor

    def _get_point_errors(self, tensor_in: torch.Tensor) -> np.ndarray:
        """
        Compute per-point reconstruction errors.
        tensor_in shape: (1, seq_len, num_features)
        Returns array of shape (num_points,) with MSE for each point.
        """
        with torch.no_grad():
            recon = self.model(tensor_in)
            # Both tensor_in and recon have shape (1, seq_len, num_features)
            # Compute MSE per point (mean over features dimension)
            point_errors = torch.mean((tensor_in - recon) ** 2, dim=2)  # (1, seq_len)
            return point_errors[0].cpu().numpy()

    def _map_to_original_points(self, flight: FlightTrack, point_errors: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Map resampled point indices back to original flight points.
        Returns top N anomalous points with lat, lon, timestamp, and score.
        """
        # Get the indices of top N points with highest errors
        top_indices = np.argsort(point_errors)[-top_n:][::-1]
        
        # Get original points sorted by time
        original_points = flight.sorted_points()
        if len(original_points) < 2:
            return []
        
        # Get time range for mapping
        t_min = original_points[0].timestamp
        t_max = original_points[-1].timestamp
        if t_max == t_min:
            return []
        
        anomaly_points = []
        for idx in top_indices:
            # Map resampled index to normalized time (0 to 1)
            progress = idx / (self.num_points - 1)
            
            # Map to original timestamp
            target_ts = t_min + progress * (t_max - t_min)
            
            # Find closest original point
            closest_point = min(original_points, key=lambda p: abs(p.timestamp - target_ts))
            
            anomaly_points.append({
                "lat": closest_point.lat,
                "lon": closest_point.lon,
                "timestamp": int(closest_point.timestamp),
                "point_score": float(point_errors[idx])
            })
        
        return anomaly_points

    def predict(self, flight: FlightTrack) -> Dict[str, Any]:
        if not self.model:
            return {"error": "Model not loaded", "anomaly_points": []}
            
        try:
            x = self.preprocess(flight)
            
            with torch.no_grad():
                loss = self.model.get_reconstruction_error(x)
                score = loss.item()
                
            is_anomaly = score > self.threshold
            
            # Extract point-level errors and find top anomalous points
            point_errors = self._get_point_errors(x)
            anomaly_points = self._map_to_original_points(flight, point_errors, top_n=5)
            
            return {
                "score": score,
                "threshold": self.threshold,
                "is_anomaly": is_anomaly,
                "severity": score / self.threshold if self.threshold > 0 else 0,
                "anomaly_points": anomaly_points
            }
        except Exception as e:
            return {"error": str(e), "anomaly_points": []}
