"""
06_Empirical_Experiment/CTGAN/generate_synthetic.py
Layer 2: Generates synthetic minority tabular records to achieve a 1:1 balance ratio.
"""

import pandas as pd
import numpy as np

def balance_training_set_with_ctgan(df_train: pd.DataFrame, ctgan_model, target_col: str = "dropout") -> pd.DataFrame:
    """
    Generates synthetic samples for class y = 1 until n_majority == n_minority.
    Appends generated samples strictly to the training set.
    """
    majority_count = (df_train[target_col] == 0).sum()
    minority_count = (df_train[target_col] == 1).sum()
    n_synthetic_needed = max(0, majority_count - minority_count)
    
    print(f"[CTGAN Synthesis] Majority: {majority_count}, Minority: {minority_count}. Generating {n_synthetic_needed} synthetic records...")
    
    if n_synthetic_needed > 0:
        synthetic_minority = ctgan_model.sample(n_synthetic_needed)
        synthetic_minority[target_col] = 1
        
        balanced_df = pd.concat([df_train, synthetic_minority], axis=0, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[CTGAN Synthesis] Balanced training set shape: {balanced_df.shape}")
        return balanced_df
    
    return df_train

if __name__ == "__main__":
    print("[CTGAN Synthesis] Synthetic generation module ready.")
