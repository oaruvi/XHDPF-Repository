"""
06_Empirical_Experiment/XGBoost/train_xgboost.py
Layer 3 (Static Branch): XGBoost Gradient Boosted Trees for Tabular SIS Predictors.
Evaluated on static administrative indicators at prediction horizon T (Term 1 completion).
"""

import numpy as np
from xgboost import XGBClassifier

def train_xgboost_static_branch(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Trains the XGBoost static branch classifier with optimized hyperparameters.
    """
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    print("[XGBoost Branch] Training complete.")
    return xgb_model

if __name__ == "__main__":
    X_dummy = np.random.randn(120, 12)
    y_dummy = np.random.choice([0, 1], 120)
    xgb_inst = train_xgboost_static_branch(X_dummy, y_dummy)
