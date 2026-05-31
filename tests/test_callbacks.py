import os
import pytest
import csv
from unittest.mock import MagicMock, patch
from src.utils.callbacks import RewardLoggerCallback, WandbAudioCallback

@pytest.fixture
def mock_model():
    """Mocks an SB3 model with an episodic information buffer."""
    model = MagicMock()
    model.ep_info_buffer = []
    return model

@pytest.fixture
def mock_env():
    """Mocks a vectorized environment with DSP attributes."""
    env = MagicMock()
    env.get_attr.return_value = [{'jitter': 0.1, 'bitrate': 64000, 'tilt': -0.5}]
    return env

def test_reward_logger_file_creation(tmp_path, mock_model):
    """Verify that headers are correctly initialized in the CSV."""
    log_dir = tmp_path / "logs"
    callback = RewardLoggerCallback(check_freq=1, id="test", log_dir=str(log_dir))
    callback.model = mock_model
    
    callback._on_training_start()
    
    save_path = log_dir / "history" / "rewards_test.csv"
    assert save_path.exists()
    
    with open(save_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ['step', 'reward', 'score', 'bonus', 'dsp_jitter', 'dsp_shimmer', 'dsp_tilt', 'dsp_harmonics', 'dsp_threshold', 'dsp_ratio', 'dsp_bitrate']

def test_reward_logger_on_step(tmp_path, mock_model):
    """Verify that rewards are appended to the CSV during training steps."""
    log_dir = tmp_path / "logs"
    callback = RewardLoggerCallback(check_freq=1, id="test", log_dir=str(log_dir))
    callback.model = mock_model
    callback.num_timesteps = 100
    callback.n_calls = 1
    
    # Mock self.locals["infos"] which is used in _on_step
    callback.locals = {
        "infos": [
            {
                "reward": 0.85, 
                "score": 0.9, 
                "bonus": 10, 
                "dsp_params": {"jitter": 0.1, "shimmer": 0.2}
            }
        ]
    }
    
    callback._on_training_start()
    callback._on_step()
    
    save_path = log_dir / "history" / "rewards_test.csv"
    with open(save_path, 'r') as f:
        lines = list(csv.reader(f))
        assert len(lines) == 2 # Header + 1 entry
        assert lines[1][0] == '100'
        assert lines[1][1] == '0.85'
        assert lines[1][2] == '0.9'
        
@patch("wandb.init")
@patch("wandb.log")
def test_wandb_audio_callback(mock_wandb_log, mock_env):
    """Verify that DSP parameters are sent to W&B at the correct frequency."""
    callback = WandbAudioCallback(config={})
    
    # training_env is a property that retrieves the env from the model.
    # We mock the model and its get_env() method instead of assigning to the property.
    callback.model = MagicMock()
    callback.model.get_env.return_value = mock_env
    
    callback.num_timesteps = 500
    
    # Case 1: n_calls is 50 (should log)
    callback.n_calls = 50
    callback._on_step()
    mock_wandb_log.assert_called_once()
    
    # Case 2: n_calls is 51 (should not log)
    mock_wandb_log.reset_mock()
    callback.n_calls = 51
    callback._on_step()
    mock_wandb_log.assert_not_called()