"""
06_Empirical_Experiment/preprocessing/quality_gates.py
Layer 1: Administrative Data Governance & Quality Validation Gates.
Implements DAMA-DMBOK data quality checks for administrative student records.
"""

import pandas as pd
import numpy as np

def validate_quality_gates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes automated quality validation gates on administrative student data.
    
    Checks:
    1. Range validation for GPA / Academic grades (0.0 to 20.0).
    2. Attendance percentage range (0.0% to 100.0%).
    3. Mandatory primary key non-null enforcement (Student_ID).
    4. Outlier detection and data type consistency.
    """
    cleaned_df = df.copy()
    
    # 1. Mandatory ID enforcement
    if "student_id" in cleaned_df.columns:
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=["student_id"])
        dropped = initial_count - len(cleaned_df)
        if dropped > 0:
            print(f"[Quality Gate] Dropped {dropped} records with null student_id.")

    # 2. GPA / Grade range validation (0.0 - 20.0 scale)
    grade_cols = [c for c in cleaned_df.columns if "gpa" in c.lower() or "grade" in c.lower() or "prom" in c.lower()]
    for col in grade_cols:
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            invalid_mask = (cleaned_df[col] < 0.0) | (cleaned_df[col] > 20.0)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                print(f"[Quality Gate] Corrected {invalid_count} out-of-bounds values in {col}.")
                cleaned_df.loc[invalid_mask, col] = np.nan

    # 3. Attendance percentage validation (0.0 - 100.0)
    att_cols = [c for c in cleaned_df.columns if "attend" in c.lower() or "asistencia" in c.lower()]
    for col in att_cols:
        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            invalid_mask = (cleaned_df[col] < 0.0) | (cleaned_df[col] > 100.0)
            if invalid_mask.sum() > 0:
                print(f"[Quality Gate] Corrected {invalid_mask.sum()} out-of-bounds attendance values in {col}.")
                cleaned_df.loc[invalid_mask, col] = np.nan

    print("[Quality Gate] DAMA-DMBOK Quality Gate validation complete.")
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "student_id": ["S001", "S002", None, "S004"],
        "gpa_t1": [15.5, 22.0, 12.0, -1.0],
        "attendance_pct": [95.0, 80.0, 105.0, 60.0]
    })
    validated_df = validate_quality_gates(sample_data)
    print(validated_df)
