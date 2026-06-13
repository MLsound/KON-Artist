import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from src.env.audio_attack import AudioAttackEnv
from src.env.wrappers import VecDetectorWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

@pytest.fixture
def mock_detector():
    """Mock detector returning a score and embedding."""
    detector = MagicMock()
    # default returns score=0.4, embedding of size 160
    detector.get_score_and_embedding.return_value = (np.array([0.4]), torch.randn(1, 160))
    return detector

@pytest.fixture
def custom_clustering_config():
    return {
        "model_path": "mock_model.pkl",
        "exclude_clusters": [5],
        "dynamic_noise": {
            "enabled": True,
            "exploration_floor": 0.3,
            "exploration_ceiling": 1.8,
            "asr_update_window": 4096
        },
        "inverse_frequency_multipliers": {0: 1.0, 1: 1.0, 5: 1.0},
        "cluster_biases": [
            [0.0] * 7,  # Cluster 0
            [0.0] * 7,  # Cluster 1
            [0.0] * 7,  # Cluster 2
            [0.0] * 7,  # Cluster 3
            [0.0] * 7,  # Cluster 4
            [0.0] * 7,  # Cluster 5
        ]
    }

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_cluster_exclusion_on_reset(mock_open, mock_pickle_load, mock_detector, custom_clustering_config):
    """
    Test that when resetting, if the GMM assigns a sample to an excluded cluster,
    the wrapper automatically triggers resets until a non-excluded cluster sample is secured.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'gmm': MagicMock(n_components=6)}
    # Let's mock predict_proba to return cluster 5 (excluded) the first time, and cluster 0 (allowed) the second time.
    # Reset is called on the batch of size 1, then _filter_excluded_clusters will call env_method("reset", indices=0)
    # which will then fetch another sample. So the first call to predict_proba will be for the initial batch.
    # Then subsequent call(s) will be for the re-evaluated reset samples.
    mock_pipeline.predict_proba.side_effect = [
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),  # First sample: 100% cluster 5 (excluded)
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])   # Second sample: 100% cluster 0 (allowed)
    ]
    mock_pickle_load.return_value = mock_pipeline

    # Create dummy base environment returning dummy wavs
    def make_base_env():
        env = MagicMock(spec=AudioAttackEnv)
        env.reset.return_value = (np.zeros(64600), {"label": 0})
        env.step.return_value = (np.zeros(64600), 0.0, False, False, {})
        env.observation_space = gym_spaces_box()
        env.action_space = gym_spaces_box_action()
        return env

    def gym_spaces_box():
        return torch.zeros(64600).numpy()  # dummy

    # Instead of full mock env, we can construct the real DummyVecEnv but mock env_method
    # or just use mock environments.
    mock_env = MagicMock()
    mock_env.reset.return_value = np.zeros((1, 64600))
    # env_method("reset", indices=0) should return [(raw_audio, info)]
    mock_env.env_method.return_value = [(np.zeros(64600), {"label": 0})]
    mock_env.num_envs = 1

    wrapped_env = VecDetectorWrapper(mock_env, mock_detector, config={"clustering": custom_clustering_config})
    
    # We call reset
    obs = wrapped_env.reset()
    
    # Assertions
    # The GMM predict_proba should have been called twice (once for initial, once for the retry)
    assert mock_pipeline.predict_proba.call_count == 2
    # env_method("reset") must have been called on worker 0
    mock_env.env_method.assert_called_with("reset", indices=0)
    # The final observation should show cluster 0 as dominant (prob = 1.0 at index 160)
    assert obs[0, 160] == 1.0

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_dynamic_noise_scaling(mock_open, mock_pickle_load, mock_detector, custom_clustering_config):
    """
    Verify that actions are contextually scaled based on cluster ASR:
    - Highly successful clusters (ASR = 1.0) -> scaled by exploration_floor (0.3)
    - Resilient clusters (ASR = 0.0) -> scaled by exploration_ceiling (1.8)
    """
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'gmm': MagicMock(n_components=6)}
    mock_pickle_load.return_value = mock_pipeline

    # Create two environment instances in DummyVecEnv
    # Environment 0 is in Cluster 0 (100% ASR)
    # Environment 1 is in Cluster 1 (0% ASR)
    mock_env = MagicMock()
    mock_env.num_envs = 2
    mock_env.reset.return_value = np.zeros((2, 64600))

    wrapped_env = VecDetectorWrapper(mock_env, mock_detector, config={"clustering": custom_clustering_config})
    
    # Populate current trajectory cluster probs
    # Env 0: dominant cluster 0
    # Env 1: dominant cluster 1
    wrapped_env.current_trajectory_cluster_probs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32)

    # Populate ASR tracker with history:
    # Cluster 0: 5 successes, 0 failures (ASR = 1.0)
    for _ in range(5):
        wrapped_env.asr_tracker.append((0, True))
    # Cluster 1: 0 successes, 5 failures (ASR = 0.0)
    for _ in range(5):
        wrapped_env.asr_tracker.append((1, False))

    # Test actions
    raw_actions = np.ones((2, 7), dtype=np.float32)

    with patch.object(mock_env, 'step_async') as mock_step_async:
        wrapped_env.step_async(raw_actions)
        sent_actions = mock_step_async.call_args[0][0]

    # Env 0 (Cluster 0, ASR=1.0) -> omega = floor = 0.3
    # Expected action: 1.0 * 0.3 = 0.3
    assert np.allclose(sent_actions[0], 0.3)

    # Env 1 (Cluster 1, ASR=0.0) -> omega = ceiling = 1.8
    # Expected action: 1.0 * 1.8 = 1.8 -> clipped to 1.0!
    assert np.allclose(sent_actions[1], 1.0)

    # Let's check raw_action and conditioned_action attributes
    assert np.allclose(wrapped_env.last_raw_actions[0], 1.0)
    assert np.allclose(wrapped_env.last_conditioned_actions[0], 0.3)
    assert np.allclose(wrapped_env.last_conditioned_actions[1], 1.0)  # clipped to 1.0
