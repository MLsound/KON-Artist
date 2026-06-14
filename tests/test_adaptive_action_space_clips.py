import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import MagicMock
from src.env.wrappers import AdaptiveActionSpaceClipsWrapper

class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(160,))
        self.success_threshold = 0.50

def test_wrapper_initialization():
    """Verify that tracking structures and parameters are correctly initialized."""
    config = {
        "clustering": {
            "asr_compression_threshold": 0.35,
            "max_compression_factor": 0.80,
            "compression_alpha": 2.0,
            "n_components": 10,
            "dynamic_noise": {
                "asr_update_window": 100
            },
            "cluster_biases": [
                [0.1] * 7 for _ in range(10)
            ]
        }
    }
    mock_env = DummyEnv()
    mock_env.success_threshold = 0.50
    
    wrapper = AdaptiveActionSpaceClipsWrapper(mock_env, config=config)
    
    assert wrapper.asr_compression_threshold == 0.35
    assert wrapper.max_compression_factor == 0.80
    assert wrapper.compression_alpha == 2.0
    assert wrapper.n_clusters == 10
    assert wrapper.asr_window_size == 100
    assert len(wrapper.asr_history) == 10
    assert wrapper.asr_history[0].shape == (100,)
    assert wrapper.cluster_biases.shape == (10, 7)
    assert np.allclose(wrapper.cluster_biases, 0.1)

def test_rolling_asr_metrics():
    """Test that Attack Success Rate is tracked and computed correctly."""
    config = {
        "clustering": {
            "n_components": 3,
            "dynamic_noise": {
                "asr_update_window": 5
            }
        }
    }
    mock_env = DummyEnv()
    mock_env.success_threshold = 0.50
    wrapper = AdaptiveActionSpaceClipsWrapper(mock_env, config=config)
    
    # Initially ASR is 0.0
    assert wrapper.get_asr(0) == 0.0
    
    # Update with some scores
    # 3 successes (>=0.50), 1 failure (<0.50)
    wrapper.update_asr_tracker(0, 0.60)
    wrapper.update_asr_tracker(0, 0.80)
    wrapper.update_asr_tracker(0, 0.30)
    wrapper.update_asr_tracker(0, 0.90)
    
    # 4 samples processed, 3 successes -> ASR = 0.75
    assert wrapper.get_asr(0) == 0.75
    
    # Update 2 more times to exceed window size (5)
    wrapper.update_asr_tracker(0, 0.20)  # failure
    wrapper.update_asr_tracker(0, 0.10)  # failure
    
    # Now history for cluster 0 contains:
    # pointer 0: 0.20 -> failure (0)
    # pointer 1: 0.10 -> failure (0)
    # pointer 2: 0.80 -> success (1)
    # pointer 3: 0.30 -> failure (0)
    # pointer 4: 0.90 -> success (1)
    # Total count = 5. Successes = 2. ASR = 2/5 = 0.40
    assert wrapper.get_asr(0) == 0.40

def test_boundary_contraction_math():
    """Verify mathematical formulation of boundary contraction."""
    # Biases baseline for cluster 1 is [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    config = {
        "clustering": {
            "asr_compression_threshold": 0.50,
            "max_compression_factor": 0.50,
            "compression_alpha": 2.0,
            "n_components": 3,
            "exclude_clusters": [2],
            "dynamic_noise": {
                "asr_update_window": 10
            },
            "cluster_biases": [
                [0.0] * 7,
                [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0] * 7
            ]
        }
    }
    
    mock_env = DummyEnv()
    mock_env.success_threshold = 0.50
    
    # Mock step to return score in info
    mock_env.step = MagicMock(return_value=(np.zeros(160), 1.0, False, False, {"score": 0.80}))
    
    wrapper = AdaptiveActionSpaceClipsWrapper(mock_env, config=config)
    
    # Set current sample cluster probs: 100% cluster 1
    wrapper.current_trajectory_cluster_probs = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    # Case A: ASR <= threshold.
    # Set ASR to 0.40 (below threshold 0.50)
    for _ in range(4):
        wrapper.update_asr_tracker(1, 0.60) # success
    for _ in range(6):
        wrapper.update_asr_tracker(1, 0.20) # failure
    assert wrapper.get_asr(1) == 0.40
    
    action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    wrapper.step(action)
    
    # Since ASR (0.40) <= threshold (0.50), lambda_c should be 1.0 (no compression).
    # action_compressed should match original action.
    mock_env.step.assert_called_once()
    called_action = mock_env.step.call_args[0][0]
    assert np.allclose(called_action, action)
    
    # Case B: ASR > threshold.
    # Set ASR to 0.75 (above threshold 0.50)
    mock_env.step.reset_mock()
    wrapper.asr_counts[1] = 0
    wrapper.asr_pointers[1] = 0
    for _ in range(3):
        wrapper.update_asr_tracker(1, 0.80) # success
    wrapper.update_asr_tracker(1, 0.20) # failure
    assert wrapper.get_asr(1) == 0.75
    
    # Delta_c = (0.75 - 0.50) / (1.0 - 0.50) = 0.50
    # Lambda_c = 1.0 - 0.50 * (0.50 ** 2) = 1.0 - 0.125 = 0.875
    # For action_i:
    # b_c = [0.5, -0.5, 0.0, ...]
    # action_i = 1.0
    # a_compressed_0 = 0.5 + 0.875 * (1.0 - 0.5) = 0.9375
    # a_compressed_1 = -0.5 + 0.875 * (1.0 - (-0.5)) = 0.8125
    # a_compressed_2 = 0.0 + 0.875 * (1.0 - 0.0) = 0.875
    
    wrapper.step(action)
    called_action_b = mock_env.step.call_args[0][0]
    expected_action = np.array([0.9375, 0.8125, 0.875, 0.875, 0.875, 0.875, 0.875], dtype=np.float32)
    assert np.allclose(called_action_b, expected_action)

def test_excluded_clusters_pass_through():
    """Verify that action bounds for excluded clusters are not compressed."""
    config = {
        "clustering": {
            "asr_compression_threshold": 0.20,
            "max_compression_factor": 0.80,
            "compression_alpha": 2.0,
            "n_components": 3,
            "exclude_clusters": [2],
            "dynamic_noise": {
                "asr_update_window": 10
            },
            "cluster_biases": [
                [0.0] * 7,
                [0.0] * 7,
                [0.5] * 7
            ]
        }
    }
    
    mock_env = DummyEnv()
    mock_env.success_threshold = 0.50
    mock_env.step = MagicMock(return_value=(np.zeros(160), 1.0, False, False, {"score": 0.80}))
    
    wrapper = AdaptiveActionSpaceClipsWrapper(mock_env, config=config)
    wrapper.current_trajectory_cluster_probs = np.array([0.0, 0.0, 1.0], dtype=np.float32) # Cluster 2 (excluded)
    
    # Set ASR high (e.g., 1.0) so it would normally compress
    wrapper.update_asr_tracker(2, 0.90)
    
    action = np.ones(7, dtype=np.float32)
    wrapper.step(action)
    
    # Action should pass through completely unchanged
    called_action = mock_env.step.call_args[0][0]
    assert np.allclose(called_action, action)
