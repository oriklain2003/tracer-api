import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from .preprocessing import TrajectoryResampler
from .clustering import TrajectoryClusterer
from .model import TrajectoryCNN

class DeepCNNDetector:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = 50
        self.num_features = 4
        
        self.clusterer = TrajectoryClusterer.load(model_dir / "clusters.joblib")
        
        with open(model_dir / "thresholds.json", "r") as f:
            self.thresholds = json.load(f)
            
        self.models = {}
        self.norms = {}
        
        for cid in self.thresholds.keys():
            # Load Norm
            with open(model_dir / f"norm_{cid}.json", "r") as f:
                self.norms[cid] = json.load(f)
                
            # Load Model
            model = TrajectoryCNN(num_features=self.num_features, seq_len=self.num_points).to(self.device)
            weight_path = model_dir / f"cnn_{cid}.pt"
            if weight_path.exists():
                model.load_state_dict(torch.load(weight_path, map_location=self.device))
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

    def predict(self, flight):
        resampler = TrajectoryResampler(num_points=self.num_points)
        df = resampler.process(flight)
        
        if df.empty:
            return {"error": "Invalid flight", "anomaly_points": []}
            
        # 1. Cluster (using flat vector)
        vec_flat = resampler.flatten(df).reshape(1, -1)
        cluster_id = str(self.clusterer.predict(vec_flat)[0])
        
        if cluster_id not in self.models:
            return {"error": "Unknown Cluster Model", "anomaly_points": []}
            
        # 2. Prepare Matrix for CNN
        mat = resampler.to_matrix(df) # [50, 4]
        
        # Normalize using Cluster Stats
        mean = np.array(self.norms[cluster_id]["mean"]) # [1, 1, 4]
        std = np.array(self.norms[cluster_id]["std"])
        
        mat_norm = (mat.reshape(1, self.num_points, self.num_features) - mean) / std
        
        # 3. Predict
        tensor_in = torch.FloatTensor(mat_norm).to(self.device)
        model = self.models[cluster_id]
        
        with torch.no_grad():
            loss = model.get_reconstruction_error(tensor_in).item()
            
        thresh = self.thresholds[cluster_id]
        is_anom = loss > thresh
        
        # 4. Extract point-level errors and find top anomalous points
        point_errors = self._get_point_errors(tensor_in, model)
        anomaly_points = self._map_to_original_points(flight, point_errors, top_n=5)
        
        return {
            "cluster_id": int(cluster_id),
            "score": loss,
            "threshold": thresh,
            "severity": loss / thresh,
            "is_anomaly": is_anom,
            "status": "ANOMALY" if is_anom else "NORMAL",
            "anomaly_points": anomaly_points
        }

