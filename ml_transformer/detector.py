import torch
import numpy as np
import joblib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from .model import TrajectoryTransformerAE
from ml_deep.preprocessing import TrajectoryResampler
from ml_deep.clustering import TrajectoryClusterer

class TransformerAnomalyDetector:
    def __init__(self, model_dir="ml_transformer/output"):
        self.model_dir = Path(model_dir)
        self.num_points = 50
        self.num_features = 4
        
        # Use the class method to load
        self.clusters = TrajectoryClusterer.load(self.model_dir / "clusters.joblib")
        
        with open(self.model_dir / "thresholds.json", "r") as f:
            self.thresholds = json.load(f)
            
        self.models = {}
        self.norms = {}
        self.resampler = TrajectoryResampler(num_points=self.num_points)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models for each cluster
        for cid in self.thresholds.keys():
            # Load Norm
            with open(self.model_dir / f"norm_{cid}.json", "r") as f:
                norm_data = json.load(f)
                self.norms[cid] = {
                    "mean": np.array(norm_data["mean"]),
                    "std": np.array(norm_data["std"])
                }
            
            # Load Model
            model = TrajectoryTransformerAE(input_dim=self.num_features, seq_len=self.num_points).to(self.device)
            model.load_state_dict(torch.load(self.model_dir / f"transformer_{cid}.pt", map_location=self.device))
            model.eval()
            self.models[cid] = model

    def _get_point_errors(self, tensor_in: torch.Tensor, model) -> np.ndarray:
        """
        Compute per-point reconstruction errors.
        tensor_in shape: (1, seq_len, num_features)
        Returns array of shape (num_points,) with MSE for each point.
        """
        with torch.no_grad():
            recon = model(tensor_in)
            # Both tensor_in and recon have shape (1, seq_len, num_features)
            # Compute MSE per point (mean over features dimension)
            point_errors = torch.mean((tensor_in - recon) ** 2, dim=2)  # (1, seq_len)
            return point_errors[0].cpu().numpy()

    def _map_to_original_points(self, flight, point_errors: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
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

    def predict(self, flight_track) -> Dict[str, Any]:
        """
        Returns dict with anomaly score, is_anomaly boolean, and anomaly_points.
        """
        # 1. Resample
        df = self.resampler.process(flight_track)
        if df.empty:
            return {"error": "Flight too short or invalid", "anomaly_points": [], "score": 0.0, "is_anomaly": False}
            
        # 2. Cluster
        vec_flat = self.resampler.flatten(df).reshape(1, -1)
        cluster_id = str(self.clusters.predict(vec_flat)[0])
        
        if cluster_id not in self.models:
            return {"error": f"No model for cluster {cluster_id}", "anomaly_points": [], "score": 0.0, "is_anomaly": False}
            
        # 3. Prepare Input
        mat = self.resampler.to_matrix(df) # (50, 4)
        norm = self.norms[cluster_id]
        
        mat_input = mat.reshape(1, self.num_points, self.num_features)
        mat_norm = (mat_input - norm["mean"]) / norm["std"]
        
        tensor_in = torch.FloatTensor(mat_norm).to(self.device)
        model = self.models[cluster_id]
        
        # 4. Inference
        with torch.no_grad():
            error = model.get_reconstruction_error(tensor_in).item()
            
        threshold = self.thresholds[cluster_id]
        is_anomaly = error > threshold
        
        # 5. Extract point-level errors and find top anomalous points
        point_errors = self._get_point_errors(tensor_in, model)
        anomaly_points = self._map_to_original_points(flight_track, point_errors, top_n=5)
        
        return {
            "cluster_id": int(cluster_id),
            "score": error,
            "threshold": threshold,
            "severity": error / threshold,
            "is_anomaly": is_anomaly,
            "status": "ANOMALY" if is_anomaly else "NORMAL",
            "anomaly_points": anomaly_points
        }

