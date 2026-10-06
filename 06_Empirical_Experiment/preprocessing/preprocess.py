"""
06_Empirical_Experiment/preprocessing/preprocess.py
Layer 1: Preprocessing, Imputation, Scaling, and Cryptographic Pseudonymization.
Implements SHA-256 student ID hashing for GDPR / DAMA-DMBOK compliance.
"""

import hashlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def pseudonymize_student_ids(df: pd.DataFrame, id_col: str = "student_id") -> pd.DataFrame:
    """
    Applies deterministic SHA-256 cryptographic hashing to Personally Identifiable Information (PII).
    """
    df_copy = df.copy()
    if id_col in df_copy.columns:
        df_copy[id_col] = df_copy[id_col].astype(str).apply(
            lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()
        )
        print(f"[Privacy] Applied SHA-256 cryptographic hashing to '{id_col}'.")
    return df_copy

def preprocess_tabular_data(df: pd.DataFrame, target_col: str = "dropout") -> tuple:
    """
    Imputes missing values (median for numerical, mode for categorical)
    and scales continuous features using StandardScaler.
    """
    df_clean = pseudonymize_student_ids(df)
    
    X = df_clean.drop(columns=[target_col, "student_id"], errors="ignore")
    y = df_clean[target_col] if target_col in df_clean.columns else None
    
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
    print(f"[Preprocessing] Processed {X.shape[1]} features across {len(X)} records.")
    return X, y

if __name__ == "__main__":
    raw_df = pd.DataFrame({
        "student_id": ["ST101", "ST102", "ST103", "ST104"],
        "gpa_t1": [14.2, np.nan, 16.8, 9.5],
        "credits_approved": [18, 12, np.nan, 6],
        "scholarship": ["Yes", "No", "No", "Yes"],
        "dropout": [0, 1, 0, 1]
    })
    X_proc, y_proc = preprocess_tabular_data(raw_df)
    print("Preprocessed Features Shape:", X_proc.shape)
