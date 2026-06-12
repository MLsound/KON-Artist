import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
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
    mock_codec.side_effect = lambda x, **kwargs: x

    env = AudioAttackEnv(mock_detector, dsp_config={}, audio_files=[(torch.randn(1, 64600), 0)])
    env.reset()

    action = np.zeros(7)
    obs, reward, terminated, _, _ = env.step(action)

    assert isinstance(reward, float)
    assert terminated is True

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_stagnation_penalty(mock_open, mock_pickle_load, mock_detector):
    """Verify that a zero-score triggers the stagnation penalty."""
    # Mock clustering pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'gmm': MagicMock(n_components=4)}
    mock_pipeline.predict_proba.return_value = np.array([[1.0, 0.0, 0.0, 0.0]])
    mock_pickle_load.return_value = mock_pipeline

    # 1. Baseline: Auxiliary reward disabled
    env_baseline = AudioAttackEnv(
        mock_detector, 
        audio_files=[(torch.randn(64600), 0)],
        clustering_config={"model_path": "mock.pkl", "auxiliary_reward": {"enabled": False}}
    )
    mock_detector.get_score_and_embedding.return_value = ([0.0], torch.randn(1, 160))
    env_baseline.reset()
    with patch.object(env_baseline.dsp, 'apply_codec_simulation', side_effect=lambda x, **kwargs: x):
        _, reward_baseline, _, _, _ = env_baseline.step(np.zeros(7))

    # 2. Penalty enabled
    clustering_config = {
        "model_path": "mock.pkl",
        "auxiliary_reward": {"enabled": True, "stagnation_penalty": -10.0, "fidelity_weight": 0.0}
    }
    env_penalty = AudioAttackEnv(
        mock_detector, 
        audio_files=[(torch.randn(64600), 0)],
        clustering_config=clustering_config
    )
    env_penalty.reset()
    with patch.object(env_penalty.dsp, 'apply_codec_simulation', side_effect=lambda x, **kwargs: x):
        _, reward_penalty, _, _, _ = env_penalty.step(np.zeros(7))

    # Should be exactly 10.0 lower than baseline (since fidelity weight is 0)
    assert np.isclose(reward_penalty, reward_baseline - 10.0)

@patch("pickle.load")
@patch("builtins.open", new_callable=MagicMock)
def test_fidelity_penalty(mock_open, mock_pickle_load, mock_detector):
    """Verify that extreme DSP parameters trigger the fidelity penalty."""
    # Mock clustering pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'gmm': MagicMock(n_components=4)}
    mock_pipeline.predict_proba.return_value = np.array([[1.0, 0.0, 0.0, 0.0]])
    mock_pickle_load.return_value = mock_pipeline

    clustering_config = {
        "model_path": "mock.pkl",
        "auxiliary_reward": {"enabled": True, "fidelity_weight": 10.0}
    }
    env = AudioAttackEnv(
        mock_detector, 
        audio_files=[(torch.randn(64600), 0)],
        clustering_config=clustering_config
    )

    mock_detector.get_score_and_embedding.return_value = ([0.5], torch.randn(1, 160))
    env.reset()

    # 1. High fidelity action
    action_hifi = np.array([-1, -1, -0.7, -0.3, 0, -1, 1]) 
    with patch.object(env.dsp, 'apply_codec_simulation', side_effect=lambda x, **kwargs: x):
        _, reward_hifi, _, _, _ = env.step(action_hifi)

    # 2. Low fidelity action
    action_lofi = np.array([-1, -1, -0.7, -0.3, 0, 1, -1])
    env.reset()
    with patch.object(env.dsp, 'apply_codec_simulation', side_effect=lambda x, **kwargs: x):
        _, reward_lofi, _, _, _ = env.step(action_lofi)

    assert reward_hifi > reward_lofi