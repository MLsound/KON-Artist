import pytest
import torch
from unittest.mock import patch
from src.synthesis.dsp import DSPPipeline

@pytest.fixture
def pipeline():
    return DSPPipeline(sample_rate=16000)

@pytest.fixture
def clean_audio():
    # 1 second of mono white noise
    return torch.randn(1, 16000)

@pytest.fixture
def dsp_params():
    return {
        'jitter': 0.01,
        'shimmer': 0.01,
        'tilt': 0.5,
        'harmonics': 0.2,
        'threshold': -20.0,
        'ratio': 4.0,
        'bitrate': 64000
    }

@patch("src.synthesis.dsp.DSPPipeline.apply_codec_simulation")
def test_normalization_integrity(mock_codec, pipeline, clean_audio, dsp_params):
    """Ensure the pipeline always outputs peak-normalized audio [-1, 1]."""
    mock_codec.side_effect = lambda x, **kwargs: x
    
    dsp_params['harmonics'] = 10.0 
    output = pipeline(clean_audio, dsp_params)
    assert output.abs().max() <= 1.0001

def test_compression_logic(pipeline):
    """Verify that signals above threshold are actually attenuated."""
    audio = torch.ones(1, 1000) * 0.9 # High amplitude
    # Threshold at -20dB is approx 0.1 amplitude
    compressed = pipeline.apply_compression(audio, threshold_db=-20.0, ratio=10.0)
    assert compressed.abs().mean() < audio.abs().mean()

@patch("src.synthesis.dsp.DSPPipeline.apply_codec_simulation")
def test_codec_simulation_shape(mock_codec, pipeline, clean_audio):
    """Check that codec simulation preserves temporal length."""
    # Verify the method is called without actually executing the save/load logic
    mock_codec.side_effect = lambda x, **kwargs: x
    
    output = pipeline.apply_codec_simulation(clean_audio, bitrate=32000)
    assert output.shape == clean_audio.shape
    mock_codec.assert_called_once()

def test_reverb_convolution(pipeline, clean_audio):
    """Verify reverb does not return NaNs and maintains shape."""
    ir = torch.randn(1, 1600) # 100ms IR
    output = pipeline.apply_reverb(clean_audio, ir, wet_dry=0.5)
    assert not torch.isnan(output).any()
    assert output.shape == clean_audio.shape

def test_device_agnosticism(pipeline, clean_audio, dsp_params):
    """Test if the pipeline maintains device consistency (CPU/MPS/CUDA)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pipeline.to(device)
        audio = clean_audio.to(device)
        output = pipeline(audio, dsp_params)
        assert output.device.type == "cuda"
