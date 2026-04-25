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
    
    return eer, eer_threshold