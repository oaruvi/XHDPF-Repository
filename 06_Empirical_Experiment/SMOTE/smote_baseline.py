"""
06_Empirical_Experiment/SMOTE/smote_baseline.py
Baseline 1: Traditional Linear Interpolation Oversampling (SMOTE).
Used as a comparative baseline against CTGAN deep generative oversampling.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def apply_smote_baseline(X_train: np.ndarray, y_train: np.ndarray, k_neighbors: int = 5) -> tuple:
    """
    Applies synthetic minority oversampling technique (SMOTE) to training data strictly within CV folds.
    """
    print(f"[SMOTE Baseline] Initial class distribution: {np.bincount(y_train)}")
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE Baseline] Resampled class distribution: {np.bincount(y_res)}")
    return X_res, y_res

def train_eval_smote_pipeline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Trains a Random Forest classifier on SMOTE-balanced training data.
    """
    X_res, y_res = apply_smote_baseline(X_train, y_train)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_res, y_res)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"[SMOTE Baseline] Test AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    return clf, auc

if __name__ == "__main__":
    np.random.seed(42)
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.choice([0, 1], size=100, p=[0.82, 0.18])
    X_tr, y_tr = X_dummy[:80], y_dummy[:80]
    X_te, y_te = X_dummy[80:], y_dummy[80:]
    train_eval_smote_pipeline(X_tr, y_tr, X_te, y_te)
