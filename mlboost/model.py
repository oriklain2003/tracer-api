from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

class XGBoostAnomalyModel:
    def __init__(self, model_path: Optional[Path] = None):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
            # scale_pos_weight will be set during fit
        )
        self.feature_names: List[str] = []
        self.model_path = model_path
        # Whether we wrapped the base estimator in a calibration layer
        self._is_calibrated: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model.
        X: DataFrame of features
        y: Series of labels (1=Anomaly, 0=Normal)
        """
        # specific handling for class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_weight = n_neg / max(1, n_pos)
        
        print(f"Training XGBoost: Normal={n_neg}, Anomaly={n_pos}, scale_pos_weight={scale_weight:.2f}")
        
        self.model.set_params(scale_pos_weight=scale_weight)
        self.feature_names = list(X.columns)
        
        self.model.fit(X, y)

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series, method: str = "isotonic"):
        """
        Wrap the trained classifier with a probability calibration layer.

        This improves the probabilistic interpretation of predict_proba outputs,
        especially under heavy class imbalance.
        """
        if len(np.unique(y_val)) < 2:
            # Cannot calibrate with a single-class validation set; skip gracefully
            print("Skipping calibration: validation set has a single class.")
            return

        print(f"Calibrating probabilities using method='{method}' on "
              f"{len(X_val)} validation samples...")

        # Use positional arguments for compatibility across sklearn versions
        calibrator = CalibratedClassifierCV(self.model, method=method, cv="prefit")
        calibrator.fit(X_val, y_val)
        self.model = calibrator
        self._is_calibrated = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Print evaluation metrics.
        """
        probs = self.predict_proba(X)
        preds = (probs > 0.5).astype(int)
        
        roc_auc = roc_auc_score(y, probs)
        precision, recall, _ = precision_recall_curve(y, probs)
        pr_auc = auc(recall, precision)
        
        # Confusion Matrix elements
        tp = np.sum((preds == 1) & (y == 1))
        tn = np.sum((preds == 0) & (y == 0))
        fp = np.sum((preds == 1) & (y == 0))
        fn = np.sum((preds == 0) & (y == 1))
        
        print("\n=== Model Evaluation ===")
        print(f"ROC AUC:       {roc_auc:.4f}")
        print(f"PR AUC:        {pr_auc:.4f}")
        print(f"Precision:     {tp / (tp + fp):.4f} (TP / (TP + FP))")
        print(f"Recall:        {tp / (tp + fn):.4f} (TP / (TP + FN))")
        print(f"False Pos Rate: {fp / (fp + tn):.4f}")
        print("\nConfusion Matrix:")
        print(f"          Pred Normal | Pred Anomaly")
        print(f"Act Normal   {tn:>6}   |   {fp:>6}")
        print(f"Act Anomaly  {fn:>6}   |   {tp:>6}")
        
        return probs

    def save(self, path: Optional[Path] = None):
        p = path or self.model_path
        if not p:
            raise ValueError("No path specified for saving.")
        
        # Ensure directory exists
        p.parent.mkdir(parents=True, exist_ok=True)
            
        state = {
            "model": self.model,
            "feature_names": self.feature_names
        }
        joblib.dump(state, p)
        print(f"Model saved to {p}")

    @classmethod
    def load(cls, path: Path) -> 'XGBoostAnomalyModel':
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
            
        state = joblib.load(path)
        instance = cls(model_path=path)
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        return instance

