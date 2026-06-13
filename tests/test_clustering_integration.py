import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from src.env.audio_attack import AudioAttackEnv
from src.env.wrappers import VecDetectorWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

@pytest.fixture
def mock_detector():
    """Mock detector returning a bonafide score and embedding."""
    detector = MagicMock()
    detector.get_score_and_embedding.return_value = ([0.95], torch.randn(1, 160))
    return detector

@pytest.fixture
def clustering_config():
    return {
        "model_path": "mock_model.pkl",
        "inverse_frequency_multipliers": {0: 10.0, 1: 1.0},
        "cluster_biases": [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # Cluster 0 bias
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5] # Cluster 1 bias
        ],
        "auxiliary_reward": {
            "enabled": True,
            "stagnation_penalty": -5.0,
            "fidelity_weight": 2.0
        }
    }

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_env_with_clustering_and_no_detector(mock_open, mock_pickle_load, clustering_config):
    """
    Test that AudioAttackEnv does NOT crash when initialized with clustering_config 
    but NO detector (worker mode).
    """
    # Mock clustering pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.25, 0.25, 0.25, 0.25]])
    mock_pickle_load.return_value = mock_pipeline
    
    audio_sample = torch.randn(64600)
    env = AudioAttackEnv(
        detector=None, 
        audio_config={"rank": 0, "seed": 42}, 
        audio_files=[(audio_sample, 0)],
        clustering_config=clustering_config
    )
    
    # Reset should return raw audio (size 64600) and NOT call clustering
    obs, info = env.reset()
    assert obs.shape == (64600,)
    mock_pipeline.predict_proba.assert_not_called()
    
    # Step should also return raw audio and NOT call clustering
    action = np.zeros(7)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (64600,)
    mock_pipeline.predict_proba.assert_not_called()

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_env_with_clustering_and_detector(mock_open, mock_pickle_load, mock_detector, clustering_config):
    """
    Test that AudioAttackEnv correctly applies clustering when a detector is present.
    """
    # Mock clustering pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'gmm': MagicMock(n_components=2)}
    mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_pickle_load.return_value = mock_pipeline
    
    audio_sample = torch.randn(64600)
    env = AudioAttackEnv(
        detector=mock_detector, 
        audio_config={"rank": 0, "seed": 42}, 
        audio_files=[(audio_sample, 0)],
        clustering_config=clustering_config
    )
    
    # Reset should return conditioned observation (162,)
    obs, info = env.reset()
    assert obs.shape == (162,)
    mock_pipeline.predict_proba.assert_called_once()
    
    # Step should also return conditioned observation without recalculating GMM probabilities
    mock_pipeline.predict_proba.reset_mock()
    action = np.zeros(7)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (162,)
    mock_pipeline.predict_proba.assert_not_called()

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_env_action_biasing(mock_open, mock_pickle_load, mock_detector, clustering_config):
    """
    Verify that AudioAttackEnv applies cluster-conditioned action biasing.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'gmm': MagicMock(n_components=2)}
    # Fixed probabilities: 100% Cluster 0
    mock_pipeline.predict_proba.return_value = np.array([[1.0, 0.0]])
    mock_pickle_load.return_value = mock_pipeline
    
    env = AudioAttackEnv(
        detector=mock_detector, 
        audio_config={"rank": 0, "seed": 42}, 
        audio_files=[(torch.randn(64600), 0)],
        clustering_config=clustering_config
    )
    env.reset()
    
    action = np.zeros(7)
    # Cluster 0 bias is [0.5, ...]
    # Expected biased action: [0.5, ...]
    with patch.object(env.dsp, 'apply_codec_simulation', side_effect=lambda x, **kwargs: x):
        env.step(action)
    
    assert np.allclose(env.last_dsp_params['jitter'], 0.5)

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_reward_multipliers(mock_open, mock_pickle_load, mock_detector, clustering_config):
    """
    Verify that compute_attack_reward applies inverse-frequency multipliers.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'gmm': MagicMock(n_components=2)}
    mock_pickle_load.return_value = mock_pipeline
    
    env = AudioAttackEnv(
        detector=mock_detector, 
        audio_config={"rank": 0, "seed": 42}, 
        audio_files=[(torch.randn(64600), 0)],
        clustering_config=clustering_config
    )
    
    # Mock score to be high so we can see the multiplier effect clearly
    mock_detector.get_score_and_embedding.return_value = ([0.9], torch.randn(1, 160))
    
    # 1. Test Cluster 0 (Multiplier 10.0)
    mock_pipeline.predict_proba.return_value = np.array([[1.0, 0.0]])
    env.reset()
    _, reward0, _, _, _ = env.step(np.zeros(7))
    
    # 2. Test Cluster 1 (Multiplier 1.0)
    mock_pipeline.predict_proba.return_value = np.array([[0.0, 1.0]])
    env.reset()
    _, reward1, _, _, _ = env.step(np.zeros(7))
    
    # Reward0 should be significantly higher than Reward1 (roughly 10x the base reward component)
    assert reward0 > reward1

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_vec_detector_wrapper_with_clustering(mock_open, mock_pickle_load, mock_detector, clustering_config):
    """
    Test that VecDetectorWrapper correctly applies clustering and action biasing.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'gmm': MagicMock(n_components=2)}
    mock_pipeline.predict_proba.return_value = np.array([[1.0, 0.0]])
    mock_pickle_load.return_value = mock_pipeline
    
    mock_detector.get_score_and_embedding.return_value = (torch.tensor([0.9]), torch.randn(1, 160))
    
    def make_base_env():
        return AudioAttackEnv(
            detector=None, 
            audio_config={"rank": 0, "seed": 42}, 
            audio_files=[(torch.randn(64600), 0)],
            clustering_config=None
        )
    
    venv = DummyVecEnv([make_base_env])
    wrapped_env = VecDetectorWrapper(venv, mock_detector, config={"clustering": clustering_config})
    
    obs = wrapped_env.reset()
    assert obs.shape == (1, 162)
    
    # Test Batched Action Biasing
    action = np.array([np.zeros(7)])
    # We need to capture the biased action passed to the underlying venv
    with patch.object(venv, 'step_async') as mock_step_async:
        wrapped_env.step_async(action)
        biased_action = mock_step_async.call_args[0][0]
        # Cluster 0 bias [0.5, ...]
        assert np.allclose(biased_action[0], 0.5)
