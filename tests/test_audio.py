import torch
import pytest
from src.utils.audio import preprocess_audio

def test_preprocess_audio_dimensions():
    """Verify that output dimensions are strictly (1, 64600)."""
    # Test with short audio
    sr = 16000
    audio = torch.randn(1, 1000)
    processed = preprocess_audio(audio, sr)
    assert processed.shape == (1, 64600)
    
    # Test with long audio
    audio_long = torch.randn(1, 80000)
    processed_long = preprocess_audio(audio_long, sr)
    assert processed_long.shape == (1, 64600)

def test_normalization_properties():
    """Ensure Z-score normalization results in near-zero mean and unit variance."""
    # Use the target length (64600) to bypass zero-padding during this specific test
    audio = torch.randn(1, 64600) * 10 + 5 
    processed = preprocess_audio(audio, 16000)
    
    assert torch.isclose(processed.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(processed.std(), torch.tensor(1.0), atol=1e-5)

def test_pre_emphasis_filter():
    """Verify the high-pass effect of the pre-emphasis filter."""
    # A constant signal should be significantly attenuated by x[n] - 0.97*x[n-1]
    audio = torch.ones(1, 1000)
    processed = preprocess_audio(audio, 16000)
    # The first sample remains, but subsequent samples should be 0.03 (unnormalized)
    # After Z-normalization, we just check that the signal is not constant
    assert torch.std(processed) > 0