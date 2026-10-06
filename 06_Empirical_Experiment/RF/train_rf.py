"""
06_Empirical_Experiment/RF/train_rf.py
Baseline 2: Random Forest Tabular Classifier.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest_baseline(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains and tunes a Random Forest model on tabular SIS features.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"[Random Forest] Best params: {grid.best_params_}")
    print(f"[Random Forest] Best Inner CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_

if __name__ == "__main__":
    X_dummy = np.random.randn(100, 8)
    y_dummy = np.random.choice([0, 1], 100)
    best_rf = train_random_forest_baseline(X_dummy, y_dummy)
