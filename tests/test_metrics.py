import numpy as np
import pytest
from src.utils.metrics import compute_eer

def test_eer_perfect_separation():
    """Verify that EER is 0.0 when scores are perfectly separated."""
    bonafide = np.array([0.9, 0.8, 0.95])
    spoof = np.array([0.1, 0.2, 0.05])
    
    eer, fpr, fnr, thresholds = compute_eer(bonafide, spoof)
    
    assert eer == 0.0

def test_eer_perfect_overlap():
    """Verify that EER is 0.5 when distributions are identical (random chance)."""
    bonafide = np.array([0.5, 0.5, 0.5])
    spoof = np.array([0.5, 0.5, 0.5])
    
    eer, _, _, _ = compute_eer(bonafide, spoof)
    
    # With the (fpr+fnr)/2 update, this will now correctly return 0.5.
    assert pytest.approx(eer) == 0.5

def test_eer_return_types():
    """Ensure the function returns the correct types and values."""
    bonafide = np.random.normal(0.7, 0.1, 100)
    spoof = np.random.normal(0.3, 0.1, 100)
    
    eer, fpr, fnr, thresholds = compute_eer(bonafide, spoof)
    
    assert isinstance(eer, (float, np.float64))
    assert 0 <= eer <= 1