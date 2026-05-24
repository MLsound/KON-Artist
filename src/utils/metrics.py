"""
Performance Evaluation Metrics for Audio Spoofing Detection.

This module provides utility functions to calculate standard classification 
metrics used in the ASVspoof challenges. Its primary function is the 
calculation of the Equal Error Rate (EER), which is the point where the 
False Acceptance Rate (FAR) and False Rejection Rate (FRR) are equal.
"""

import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray):
    """
    Calculates the Equal Error Rate (EER) following the ASVspoof standard.
    
    The EER is computed by finding the point on the ROC curve where 
    the False Positive Rate (FPR) and False Negative Rate (FNR) intersect.
    """
    # Labels: 1 for bonafide, 0 for spoof
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    
    # Generate ROC curve points
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    # False Negative Rate (FNR) is 1 - True Positive Rate (TPR)
    fnr = 1 - tpr
    
    # Find the index where the difference between FPR and FNR is minimized
    idx = np.nanargmin(np.absolute(fnr - fpr))
    
    # Calculate EER as the average of FPR and FNR at the crossover point.
    # This handles edge cases like identical distributions (0.5 EER) correctly.
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]
    
    return eer, fpr, fnr, thresholds

def compute_mindcf(frr: np.ndarray, far: np.ndarray, thresholds: np.ndarray, 
                  Pspoof: float = 0.05, Cmiss: float = 1.0, Cfa: float = 10.0):
    """
    Calculates the Minimum Detection Cost Function (min DCF).
    Commonly used in ASVspoof as a secondary metric to EER.
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]

    p_target = 1 - Pspoof
    for i in range(len(frr)):
        # Weighted sum of false negative and false positive errors
        c_det = Cmiss * frr[i] * p_target + Cfa * far[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
            
    # Normalize the cost
    c_def = min(Cmiss * p_target, Cfa * (1 - p_target))
    min_dcf = min_c_det / c_def
    
    return min_dcf, min_c_det_threshold

def calculate_cllr(bonafide_scores: np.ndarray, spoof_scores: np.ndarray):
    """
    Calculates the Cost of Log-Likelihood Ratio (CLLR).
    Measures the well-calibratedness of the detector scores.
    """
    def negative_log_sigmoid(lodds):
        return np.log1p(np.exp(-lodds))

    # Calculate the CLLR value
    cllr = 0.5 * (np.mean(negative_log_sigmoid(bonafide_scores)) + \
                  np.mean(negative_log_sigmoid(-spoof_scores))) / np.log(2)

    return cllr
