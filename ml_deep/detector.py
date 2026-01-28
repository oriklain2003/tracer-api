import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from .preprocessing import TrajectoryResampler
from .clustering import TrajectoryClusterer
from .model import TrajectoryAutoencoder

class DeepAnomalyDetector:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = 50
        self.num_features = 4  # lat, lon, alt, track
        
        # Load Clustering Model
        self.clusterer = TrajectoryClusterer.load(model_dir / "clusters.joblib")
        
        # Load Thresholds
        with open(model_dir / "thresholds.json", "r") as f:
            self.thresholds = json.load(f)
            
        # Load Autoencoders (Lazy Load or Preload all)
        self.aes = {}
        for cid in self.thresholds.keys():
            # Initialize Model Structure
            # We need to know input dimension. 
            # We can get it from the scaler mean shape in the clusterer
            input_dim = self.clusterer.scaler.mean_.shape[0]
            
            model = TrajectoryAutoencoder(input_dim=input_dim).to(self.device)
            
            # Load Weights
            weight_path = model_dir / f"ae_cluster_{cid}.pt"
            if weight_path.exists():
                model.load_state_dict(torch.load(weight_path, map_location=self.device))
                model.eval()
                self.aes[cid] = model
            else:
                print(f"Warning: Model for cluster {cid} not found at {weight_path}")

    def _get_point_errors(self, tensor_in: torch.Tensor, model) -> np.ndarray:
        """
        Compute per-point reconstruction errors.
        Returns array of shape (num_points,) with MSE for each point.
        """
        with torch.no_grad():
            recon = model(tensor_in)
            # tensor_in and recon are shape (1, num_points * num_features)
            # Reshape to (num_points, num_features)
            input_reshaped = tensor_in.view(self.num_points, self.num_features)
            recon_reshaped = recon.view(self.num_points, self.num_features)
            
            # Compute MSE per point (mean over features)
            point_errors = torch.mean((input_reshaped - recon_reshaped) ** 2, dim=1)
            return point_errors.cpu().numpy()

    def _map_to_original_points(self, flight, resampled_df, point_errors: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
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
        # 1. Preprocess
        resampler = TrajectoryResampler(num_points=self.num_points)
        df = resampler.process(flight)
        
        if df.empty:
            return {"error": "Flight too short or invalid", "anomaly_points": []}
            
        vec = resampler.flatten(df)
        vec_reshaped = vec.reshape(1, -1) # Shape (1, 200)
        
        # 2. Determine Flow (Cluster)
        cluster_id = self.clusterer.predict(vec_reshaped)[0]
        cluster_id_str = str(cluster_id)
        
        if cluster_id_str not in self.aes:
            return {"error": f"No model for cluster {cluster_id}", "anomaly_points": []}
            
        # 3. Check Anomaly (Autoencoder)
        model = self.aes[cluster_id_str]
        
        # Normalize input (using same scaler as training!)
        vec_norm = self.clusterer.scaler.transform(vec_reshaped)
        tensor_in = torch.FloatTensor(vec_norm).to(self.device)
        
        with torch.no_grad():
            loss = model.get_reconstruction_error(tensor_in).item()
            
        threshold = self.thresholds[cluster_id_str]
        is_anomaly = loss > threshold
        
        # Calculate Severity (Score / Threshold)
        severity = loss / threshold
        
        # 4. Extract point-level errors and find top anomalous points
        point_errors = self._get_point_errors(tensor_in, model)
        anomaly_points = self._map_to_original_points(flight, df, point_errors, top_n=5)
        
        return {
            "cluster_id": int(cluster_id),
            "score": loss,
            "threshold": threshold,
            "severity": severity,
            "is_anomaly": is_anomaly,
            "status": "ANOMALY" if is_anomaly else "NORMAL",
            "anomaly_points": anomaly_points
        }

