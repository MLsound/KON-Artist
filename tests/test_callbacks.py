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
        assert header == [
            'step', 'session_step', 'session_progress_pct', 'reward', 'score', 'bonus', 'cluster', 
            'dsp_jitter', 'dsp_shimmer', 'dsp_tilt', 'dsp_harmonics', 'dsp_threshold', 'dsp_ratio', 'dsp_bitrate',
            'raw_action_0', 'raw_action_1', 'raw_action_2', 'raw_action_3', 'raw_action_4', 'raw_action_5', 'raw_action_6',
            'conditioned_action_0', 'conditioned_action_1', 'conditioned_action_2', 'conditioned_action_3', 'conditioned_action_4', 'conditioned_action_5', 'conditioned_action_6'
        ]

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
        assert lines[1][1] == '100'
        assert lines[1][2] == '0.0'
        assert lines[1][3] == '0.85'
        assert lines[1][4] == '0.9'
        
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

def test_entropy_decay_callback_global():
    """Verify that global mode calculates decay relative to cumulative total steps."""
    from src.utils.callbacks import EntropyDecayCallback
    
    mock_model = MagicMock()
    callback = EntropyDecayCallback(
        initial_ent_coef=0.1,
        final_ent_coef=0.01,
        total_timesteps=1000,
        initial_checkpoint_steps=500,
        session_total_steps=500,
        mode="global"
    )
    callback.model = mock_model
    
    # At cumulative step 750 (which is 250 steps in session)
    callback.num_timesteps = 750
    callback._on_step()
    
    # Progress = 750 / 1000 = 0.75
    # ent_coef = 0.1 + (0.01 - 0.1) * 0.75 = 0.1 - 0.09 * 0.75 = 0.0325
    assert mock_model.ent_coef == pytest.approx(0.0325)

def test_entropy_decay_callback_cycle():
    """Verify that cycle mode calculates decay relative to current cycle steps."""
    from src.utils.callbacks import EntropyDecayCallback
    
    mock_model = MagicMock()
    callback = EntropyDecayCallback(
        initial_ent_coef=0.1,
        final_ent_coef=0.01,
        total_timesteps=1000,
        initial_checkpoint_steps=500,
        session_total_steps=500,
        mode="cycle"
    )
    callback.model = mock_model
    
    # At cumulative step 750 (which is 250 steps in session)
    callback.num_timesteps = 750
    callback._on_step()
    
    # Progress = (750 - 500) / 500 = 0.5
    # ent_coef = 0.1 + (0.01 - 0.1) * 0.5 = 0.1 - 0.09 * 0.5 = 0.055
    assert mock_model.ent_coef == pytest.approx(0.055)


def test_cluster_manifold_alignment_callback():
    """Verify that ClusterManifoldAlignmentCallback correctly tracks, predicts and logs action shift metrics."""
    from src.utils.callbacks import ClusterManifoldAlignmentCallback
    import numpy as np
    
    # Mock model
    mock_model = MagicMock()
    # Observation space of size 170
    mock_model.observation_space.shape = (170,)
    
    # Mock policy predict
    # actions should have shape (sample_size, 7)
    # Let's mock a fixed action pair: tilt = -0.8, harmonics = 0.9
    sample_size = 10
    mock_actions = np.zeros((sample_size, 7))
    mock_actions[:, 2] = -0.8
    mock_actions[:, 3] = 0.9
    mock_model.policy.predict.return_value = (mock_actions, None)
    
    # Mock logger
    mock_logger = MagicMock()
    
    # Mock env
    mock_eval_env = MagicMock()
    
    callback = ClusterManifoldAlignmentCallback(
        eval_env=mock_eval_env,
        target_clusters=[3, 9],
        log_freq_updates=2,
        sample_size=sample_size
    )
    mock_model.logger = mock_logger
    callback.model = mock_model
    assert callback.logger is mock_logger
    
    # Rollout 1: self.rollout_counter becomes 1. Skip (1 % 2 != 0)
    callback._on_rollout_end()
    assert callback.rollout_counter == 1
    mock_model.policy.predict.assert_not_called()
    
    # Rollout 2: self.rollout_counter becomes 2. Run (2 % 2 == 0)
    callback._on_rollout_end()
    assert callback.rollout_counter == 2
    
    # Predict should have been called twice (once for cluster 3, once for cluster 9)
    assert mock_model.policy.predict.call_count == 2
    
    # Verify the observation input to predict
    call_args_list = mock_model.policy.predict.call_args_list
    # Cluster 3 prediction check
    obs_cluster_3 = call_args_list[0][0][0]
    assert obs_cluster_3.shape == (sample_size, 170)
    # Check that GMM representation is one-hot at index 163
    assert np.allclose(obs_cluster_3[:, 163], 1.0)
    assert np.allclose(obs_cluster_3[:, :160], 0.0)
    
    # Cluster 9 prediction check
    obs_cluster_9 = call_args_list[1][0][0]
    assert obs_cluster_9.shape == (sample_size, 170)
    # Check that GMM representation is one-hot at index 169
    assert np.allclose(obs_cluster_9[:, 169], 1.0)
    
    # Verify distance computation
    # Target anchor point: tilt = -1.0, harmonics = 1.0
    # Mean tilt = -0.8, Mean harmonics = 0.9
    # Distance = (-0.8 - (-1.0))**2 + (0.9 - 1.0)**2 = (0.2)**2 + (-0.1)**2 = 0.04 + 0.01 = 0.05
    mock_logger.record.assert_any_call("manifold_alignment/cluster_3_tilt_mean", -0.8)
    mock_logger.record.assert_any_call("manifold_alignment/cluster_3_harmonics_mean", 0.9)
    mock_logger.record.assert_any_call("manifold_alignment/cluster_3_distance_to_target", pytest.approx(0.05))
    
    mock_logger.record.assert_any_call("manifold_alignment/cluster_9_tilt_mean", -0.8)
    mock_logger.record.assert_any_call("manifold_alignment/cluster_9_harmonics_mean", 0.9)
    mock_logger.record.assert_any_call("manifold_alignment/cluster_9_distance_to_target", pytest.approx(0.05))