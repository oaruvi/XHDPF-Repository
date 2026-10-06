import numpy as np
import pandas as pd

def calculate_disparate_impact(y_pred, protected_attr, unprivileged_val=1, privileged_val=0):
    """
    Computes Disparate Impact (DI) ratio:
    DI = P(hat{y}=1 | unprivileged) / P(hat{y}=1 | privileged)
    """
    unprivileged_mask = (protected_attr == unprivileged_val)
    privileged_mask = (protected_attr == privileged_val)
    
    rate_unprivileged = np.mean(y_pred[unprivileged_mask])
    rate_privileged = np.mean(y_pred[privileged_mask])
    
    if rate_privileged == 0:
        return np.nan
    return rate_unprivileged / rate_privileged

def calculate_equalized_odds_difference(y_true, y_pred, protected_attr, unprivileged_val=1, privileged_val=0):
    """
    Computes Equalized Odds Difference (Delta EOD):
    Delta EOD = |TPR_unpriv - TPR_priv| + |FPR_unpriv - FPR_priv|
    """
    unprivileged_mask = (protected_attr == unprivileged_val)
    privileged_mask = (protected_attr == privileged_val)
    
    # True Positives & False Positives for Unprivileged
    tp_unpriv = np.sum((y_pred[unprivileged_mask] == 1) & (y_true[unprivileged_mask] == 1))
    fn_unpriv = np.sum((y_pred[unprivileged_mask] == 0) & (y_true[unprivileged_mask] == 1))
    fp_unpriv = np.sum((y_pred[unprivileged_mask] == 1) & (y_true[unprivileged_mask] == 0))
    tn_unpriv = np.sum((y_pred[unprivileged_mask] == 0) & (y_true[unprivileged_mask] == 0))
    
    # True Positives & False Positives for Privileged
    tp_priv = np.sum((y_pred[privileged_mask] == 1) & (y_true[privileged_mask] == 1))
    fn_priv = np.sum((y_pred[privileged_mask] == 0) & (y_true[privileged_mask] == 1))
    fp_priv = np.sum((y_pred[privileged_mask] == 1) & (y_true[privileged_mask] == 0))
    tn_priv = np.sum((y_pred[privileged_mask] == 0) & (y_true[privileged_mask] == 0))
    
    tpr_unpriv = tp_unpriv / (tp_unpriv + fn_unpriv) if (tp_unpriv + fn_unpriv) > 0 else 0
    tpr_priv = tp_priv / (tp_priv + fn_priv) if (tp_priv + fn_priv) > 0 else 0
    
    fpr_unpriv = fp_unpriv / (fp_unpriv + tn_unpriv) if (fp_unpriv + tn_unpriv) > 0 else 0
    fpr_priv = fp_priv / (fp_priv + tn_priv) if (fp_priv + tn_priv) > 0 else 0
    
    delta_tpr = abs(tpr_unpriv - tpr_priv)
    delta_fpr = abs(fpr_unpriv - fpr_priv)
    
    delta_eod = delta_tpr + delta_fpr
    return delta_eod, tpr_unpriv, tpr_priv, fpr_unpriv, fpr_priv

if __name__ == '__main__':
    print('Fairness Metrics Computation Module loaded successfully.')
