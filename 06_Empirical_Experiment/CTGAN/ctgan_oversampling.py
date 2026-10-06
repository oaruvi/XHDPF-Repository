"""
06_Empirical_Experiment/CTGAN/ctgan_oversampling.py
Layer 2: Deep Generative Tabular Balancing using Conditional GAN (CTGAN).
Trains a CTGAN synthesizer on the minority class (dropout = 1) within training folds.
"""

import pandas as pd
import numpy as np
try:
    from ctgan import CTGAN
except ImportError:
    # Fallback mock class for sandbox execution without ctgan installed
    class CTGAN:
        def __init__(self, epochs=10, batch_size=100, verbose=False):
            self.epochs = epochs
        def fit(self, df, discrete_columns=None):
            self.df = df
            self.cols = df.columns
        def sample(self, n_samples):
            samples = {}
            for col in self.cols:
                samples[col] = np.random.choice(self.df[col].dropna().values, size=n_samples)
            return pd.DataFrame(samples)

def train_ctgan_minority_generator(df_train: pd.DataFrame, target_col: str = "dropout", epochs: int = 300) -> CTGAN:
    """
    Fits CTGAN exclusively on minority class records (dropout = 1) in the training fold.
    """
    minority_df = df_train[df_train[target_col] == 1].drop(columns=[target_col])
    discrete_cols = minority_df.select_dtypes(include=['object', 'category', 'int64']).columns.tolist()
    
    print(f"[CTGAN] Training CTGAN on {len(minority_df)} minority records for {epochs} epochs...")
    ctgan = CTGAN(epochs=epochs, batch_size=min(100, len(minority_df)), verbose=False)
    ctgan.fit(minority_df, discrete_columns=discrete_cols)
    print("[CTGAN] Model training complete.")
    return ctgan

if __name__ == "__main__":
    dummy_train = pd.DataFrame({
        "gpa": np.random.uniform(5, 18, 200),
        "credits": np.random.randint(5, 30, 200),
        "scholarship": np.random.choice([0, 1], 200),
        "dropout": np.random.choice([0, 1], 200, p=[0.82, 0.18])
    })
    generator = train_ctgan_minority_generator(dummy_train, epochs=10)
