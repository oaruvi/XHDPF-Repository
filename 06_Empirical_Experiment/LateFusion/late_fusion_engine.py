"""
06_Empirical_Experiment/LateFusion/late_fusion_engine.py
Layer 3: Late Fusion Engine (CTGAN + XGBoost + Stacked LSTM).
Combines static background risk P_XGBoost with dynamic trajectory risk P_LSTM.
"""

import numpy as np

class LateFusionEngine:
    """
    Late Fusion Engine combining static tabular prediction and dynamic sequential prediction.
    Formula: P_final = alpha * P_XGBoost + (1 - alpha) * P_LSTM
    Optimal weight: alpha = 0.60
    """
    def __init__(self, alpha: float = 0.60):
        self.alpha = alpha

    def predict_proba(self, p_static: np.ndarray, p_dynamic: np.ndarray) -> np.ndarray:
        """
        Calculates weighted probability fusion.
        """
        p_static = np.asarray(p_static).ravel()
        p_dynamic = np.asarray(p_dynamic).ravel()
        
        p_final = self.alpha * p_static + (1.0 - self.alpha) * p_dynamic
        return p_final

if __name__ == "__main__":
    engine = LateFusionEngine(alpha=0.60)
    p_xgb = np.array([0.80, 0.20, 0.95])
    p_lstm = np.array([0.70, 0.30, 0.85])
    p_fused = engine.predict_proba(p_xgb, p_lstm)
    print("Fused Dropout Probabilities:", p_fused)
