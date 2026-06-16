import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock
from src.env.wrappers import AdaptiveActionSpaceClipsWrapper

class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(160,), dtype=np.float32)
        self.success_threshold = 0.50

def test_global_wrapper_initialization():
    """Verify that tracking structures and parameters are correctly initialized globally."""
    config = {
        "clipping": {
            "asr_compression_threshold": 0.35,
            "max_compression_factor": 0.80,
            "compression_alpha": 2.0,
            "asr_update_window": 100
        }
    }
    mock_env = DummyEnv()
    mock_env.success_threshold = 0.50
    
    wrapper = AdaptiveActionSpaceClipsWrapper(mock_env, config=config)
    
    assert wrapper.asr_compression_threshold == 0.35
    assert wrapper.max_compression_factor == 0.80
    assert wrapper.compression_alpha == 2.0
    assert wrapper.asr_window_size == 100
    assert wrapper.asr_history.shape == (100,)
    assert wrapper.successful_actions_buffer.shape == (100, 7)
    assert wrapper.asr_count == 0
    assert wrapper.successful_actions_count == 0

def test_global_rolling_asr_and_successful_actions_tracking():
    """Test that global Attack Success Rate and successful actions are tracked and averaged correctly."""
    config = {
        "clipping": {
            "asr_update_window": 5
        }
    }
    mock_env = DummyEnv()
    mock_env.success_threshold = 0.50
    wrapper = AdaptiveActionSpaceClipsWrapper(mock_env, config=config)
    
    # Initially ASR is 0.0, and successful actions average is all zeros
    assert wrapper.get_asr() == 0.0
    assert np.allclose(wrapper.get_successful_actions_average(), 0.0)
    
    # Update with some scores and actions
    # Success 1: score = 0.60, action = [1, 1, 1, 1, 1, 1, 1]
    wrapper.update_trackers(0.60, np.ones(7, dtype=np.float32))
    # Success 2: score = 0.80, action = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    wrapper.update_trackers(0.80, np.ones(7, dtype=np.float32) * 0.5)
    # Failure 1: score = 0.30, action = [-1, -1, -1, -1, -1, -1, -1] (should NOT be tracked in successful actions)
    wrapper.update_trackers(0.30, -np.ones(7, dtype=np.float32))
    # Success 3: score = 0.90, action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    wrapper.update_trackers(0.90, np.zeros(7, dtype=np.float32))
    
    # 4 samples processed, 3 successes -> ASR = 0.75
    assert wrapper.get_asr() == 0.75
    
    # Successful actions average should be mean([ones(7), ones(7)*0.5, zeros(7)]) = ones(7) * 0.5
    expected_avg = np.ones(7, dtype=np.float32) * 0.5
    assert np.allclose(wrapper.get_successful_actions_average(), expected_avg)
    
    # Push out oldest samples by updating 2 more times to exceed window size (5)
    wrapper.update_trackers(0.20, np.zeros(7, dtype=np.float32))  # failure
    wrapper.update_trackers(0.10, np.zeros(7, dtype=np.float32))  # failure
    
    # Now history contains:
    # ASR: 3 successes out of 5 -> 0.40
    assert wrapper.get_asr() == 0.40

def test_global_boundary_contraction_math():
    """Verify mathematical formulation of boundary contraction anchoring to global mean of successful actions."""
    config = {
        "clipping": {
            "asr_compression_threshold": 0.50,
            "max_compression_factor": 0.50,
            "compression_alpha": 2.0,
            "asr_update_window": 10
        }
    }
    
    mock_env = DummyEnv()
    mock_env.success_threshold = 0.50
    
    # Mock step to return score in info
    mock_env.step = MagicMock(return_value=(np.zeros(160), 1.0, False, False, {"score": 0.80}))
    
    wrapper = AdaptiveActionSpaceClipsWrapper(mock_env, config=config)
    
    # Case A: ASR <= threshold.
    # Set ASR to 0.40 (below threshold 0.50)
    for _ in range(4):
        wrapper.update_trackers(0.60, np.ones(7, dtype=np.float32) * 0.5) # success
    for _ in range(6):
        wrapper.update_trackers(0.20, -np.ones(7, dtype=np.float32)) # failure
    assert wrapper.get_asr() == 0.40
    
    action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    wrapper.step(action)
    
    # Since ASR (0.40) <= threshold (0.50), lambda_val should be 1.0 (no compression).
    # action_compressed should match original action.
    mock_env.step.assert_called_once()
    called_action = mock_env.step.call_args[0][0]
    assert np.allclose(called_action, action)
    
    # Case B: ASR > threshold.
    # Set ASR to 0.75 (above threshold 0.50)
    mock_env.step.reset_mock()
    wrapper.asr_count = 0
    wrapper.asr_pointer = 0
    wrapper.successful_actions_count = 0
    wrapper.successful_actions_pointer = 0
    
    # 3 successes with action [0.5, 0.5, ...], 1 failure
    for _ in range(3):
        wrapper.update_trackers(0.80, np.ones(7, dtype=np.float32) * 0.5) # success
    wrapper.update_trackers(0.20, -np.ones(7, dtype=np.float32)) # failure
    
    assert wrapper.get_asr() == 0.75
    # anchor is [0.5, 0.5, ...]
    assert np.allclose(wrapper.get_successful_actions_average(), np.ones(7, dtype=np.float32) * 0.5)
    
    # Delta = (0.75 - 0.50) / (1.0 - 0.50) = 0.50
    # Lambda = 1.0 - 0.50 * (0.50 ** 2) = 1.0 - 0.125 = 0.875
    # For action_i = 1.0:
    # anchor_i = 0.5
    # a_compressed = 0.5 + 0.875 * (1.0 - 0.5) = 0.9375
    
    wrapper.step(action)
    called_action_b = mock_env.step.call_args[0][0]
    expected_action = np.ones(7, dtype=np.float32) * 0.9375
    assert np.allclose(called_action_b, expected_action)
