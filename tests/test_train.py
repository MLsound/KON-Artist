import pytest
from unittest.mock import patch, MagicMock
from scripts.train import train
import logging

@pytest.fixture
def mock_cfg():
    return {
        'ppo': {
            'total_updates': 1,
            'n_steps': 10,
            'epochs': 1,
            'batch_size': 5,
            'learning_rate': 0.0001,
            'policy': 'MlpPolicy',
            'verbose': 0
        },
        'env': {
            'n_envs': 1,
            'seed': 42,
            'success_threshold': 0.5,
            'step_limit': 10
        },
        'audio': {
            'split': 'train',
            'buffer_size': 10
        },
        'model': {
            'detector_name': 'MTUCI/AASIST3',
            'device': 'cpu'
        },
        'logging': {
            'log_dir': 'outputs/',
            'reward_log_freq': 1,
            'time_per_step': 0.1,
            'checkpoint': {'load_last': False},
            'tensorboard_log': None
        }
    }

@patch("scripts.train.AASISTWrapper")
@patch("scripts.train.AudioAttackEnv")
@patch("scripts.train.SubprocVecEnv")
@patch("scripts.train.DummyVecEnv")
@patch("scripts.train.VecDetectorWrapper")
@patch("scripts.train.VecMonitor")
@patch("scripts.train.PPO")
@patch("scripts.train.RewardLoggerCallback")
@patch("scripts.train.WandbAudioCallback") 
def test_train_flow(
    mock_wandb_cb, mock_reward_cb, mock_ppo, mock_vec_monitor,
    mock_vec_wrapper, mock_dummy_vec, mock_subproc_vec, mock_env, mock_wrapper, mock_cfg
):
    """
    Verify that the training function correctly initializes all 
    components and triggers the learning process.
    """
    import scripts.train
    with patch.object(scripts.train, "cfg", mock_cfg), \
         patch.object(scripts.train, "N_ENVS", 1), \
         patch.object(scripts.train, "TOTAL_ROLLOUT_BUFFER", 10), \
         patch.object(scripts.train, "TOTAL_TIMESTEPS", 10), \
         patch.object(scripts.train, "BATCH_SIZE", 5):
        
        mock_ppo_instance = mock_ppo.return_value
        
        train()
        
        mock_wrapper.assert_called_once_with('MTUCI/AASIST3', device='cpu')
        mock_ppo_instance.learn.assert_called_once()

@patch("scripts.train.AASISTWrapper")
def test_train_error_handling(mock_wrapper, mock_cfg, caplog):
    """Ensure exceptions during training are logged but don't crash the script."""
    import scripts.train
    with patch.object(scripts.train, "cfg", mock_cfg):
        mock_wrapper.side_effect = Exception("Model weight error")
        
        # Access the specific logger instance and force propagation for caplog
        import logging
        target_logger = logging.getLogger("train")
        target_logger.propagate = True
        
        with caplog.at_level(logging.ERROR, logger="train"):
            train()
            
        assert "Training failed with a critical error" in caplog.text
        target_logger.propagate = False