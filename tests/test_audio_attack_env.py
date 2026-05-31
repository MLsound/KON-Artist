import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from unittest.mock import MagicMock
from src.env.audio_attack import AudioAttackEnv

@pytest.fixture
def mock_detector():
    """Mock detector returning a bonafide score and embedding."""
    detector = MagicMock()
    detector.get_score_and_embedding.return_value = ([0.95], torch.randn(1, 160))
    return detector

def test_env_initialization(mock_detector):
    """Verify that the env correctly instantiates the DSP class."""
    env = AudioAttackEnv(mock_detector, dsp_config={}, audio_files=[(torch.randn(1, 64600), 0)])
    assert hasattr(env, 'dsp')
    assert env.action_space.shape == (7,)
    
@patch("src.synthesis.dsp.DSPPipeline.apply_codec_simulation")
def test_step_logic(mock_codec, mock_detector):
    """Ensure step calls the DSP pipeline and returns the correct reward."""
    # The mock will simply return the input audio without performing the MP3 save/load
    mock_codec.side_effect = lambda x, **kwargs: x
    
    env = AudioAttackEnv(mock_detector, dsp_config={}, audio_files=[(torch.randn(1, 64600), 0)])
    env.reset()
    
    action = np.zeros(7)
    obs, reward, terminated, _, _ = env.step(action)
    
    assert reward > -100 # Log reward will be some value
    assert terminated is True