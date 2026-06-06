"""
Performance Evaluation Metrics for Audio Spoofing Detection.

This module provides utility functions to calculate standard classification 
metrics used in the ASVspoof challenges. Its primary function is the 
calculation of the Equal Error Rate (EER), which is the point where the 
False Acceptance Rate (FAR) and False Rejection Rate (FRR) are equal.
"""

import numpy as np
from sklearn.metrics import roc_curve

def compute_det_curve(bonafide_scores, spoof_scores):
    """
    Computes FRR and FAR with their thresholds.
    """
    all_scores = np.concatenate((bonafide_scores, spoof_scores))
    labels = np.concatenate((np.ones(bonafide_scores.size), np.zeros(spoof_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = spoof_scores.size - (np.arange(1, all_scores.size + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / bonafide_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / spoof_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds

def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray):
    """
    Calculates the Equal Error Rate (EER) following the ASVspoof standard.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, far, frr, thresholds

def compute_tDCF(bonafide_scores_cm, spoof_scores_cm, 
                 Pfa_asv=0.0243, Pmiss_asv=0.00, Pmiss_spoof_asv=0.7628):
    """
    Computes the Tandem Detection Cost Function (t-DCF) using ASVspoof 2019 LA constants.
    
    Default ASV error rates are from the 2019 LA evaluation set.
    """
    # Official ASVspoof 2019 LA constants
    cost_model = {
        'Ptar': 0.9405,
        'Pnon': 0.0595,
        'Pspoof': 0.05,
        'Cmiss_asv': 1.0,
        'Cfa_asv': 10.0,
        'Cmiss_cm': 1.0,
        'Cfa_cm': 10.0
    }

    Pmiss_cm, Pfa_cm, _ = compute_det_curve(bonafide_scores_cm, spoof_scores_cm)

    # Constants C1 and C2 for the linear combination of CM errors
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv

    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1.0 - Pmiss_spoof_asv)

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF (min t-DCF is the minimum of this curve)
    tDCFnorm = tDCF / np.minimum(C1, C2)
    
    return np.min(tDCFnorm)

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
    # Note: ASVspoof scores are often log-likelihoods or similar.
    # If they are probabilities, we should convert them to log-odds.
    # Here we assume they are calibrated scores where higher = more bonafide.
    cllr = 0.5 * (np.mean(negative_log_sigmoid(bonafide_scores)) + \
                  np.mean(negative_log_sigmoid(-spoof_scores))) / np.log(2)

    return cllr
