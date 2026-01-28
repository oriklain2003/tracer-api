from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

class TrajectoryClusterer:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray):
        # X shape: (N_flights, K_points * N_features)
        print(f"Fitting scaler on {X.shape}...")
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
        
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kmeans": self.kmeans, "scaler": self.scaler}, path)
        
    @classmethod
    def load(cls, path: Path) -> 'TrajectoryClusterer':
        data = joblib.load(path)
        instance = cls(n_clusters=data["kmeans"].n_clusters)
        instance.kmeans = data["kmeans"]
        instance.scaler = data["scaler"]
        return instance

