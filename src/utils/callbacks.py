"""
Training Callbacks for Reinforcement Learning (KON-Artist Project).

This module provides custom Stable Baselines3 (SB3) callbacks to monitor 
and log training progress. It includes utilities for local CSV reward 
logging—necessary for convergence plotting—and Weights & Biases (W&B) 
integration for tracking real-time DSP parameter evolution.
"""
import os
import csv
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.DEBUG)

class RewardLoggerCallback(BaseCallback):
    """
    Callback to save episodic rewards into a CSV file.
    Essential for generating convergence plots for final reports.
    """
    def __init__(self, check_freq: int, id: str, log_dir: str, verbose: int = 1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.id = id
        self.save_path = os.path.join(log_dir, f'history/rewards_{id}.csv')
        
        # Create directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        """Initialize the CSV file with headers at the start of training."""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'reward'])
        logger.info(f"Reward logging started at: {self.save_path}")

    def _on_step(self) -> bool:
        """
        Log the last episodic reward from the model buffer at a fixed frequency.
        SB3 stores the cumulative reward in 'ep_info_buffer'.
        """
        if self.n_calls % self.check_freq == 0:
            # Extract the last reward if available in the environment buffer
            if len(self.model.ep_info_buffer) > 0:
                last_reward = self.model.ep_info_buffer[-1]['r']
                with open(self.save_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.num_timesteps, last_reward])
                
                if self.verbose > 0:
                    logger.debug(f"Step {self.num_timesteps}: Reward saved: {last_reward}")
        return True


class WandbAudioCallback(BaseCallback):
    """
    Callback to log environment-specific DSP parameters to Weights & Biases.
    """
    # 1. Start the W&B session (Required for WandbAudioCallback to function)
    def __init__(self, config: dict, verbose=0):
        super().__init__(verbose)
        # Initialize wandb here, or pass an active run instance
        if wandb.run is None:
            wandb.init(
                entity="ml-sound",
                project="kon-artist",
                notes="Testing AASIST3 attack boundaries",
                config={**config, "system": "MLsound"} # Track hyperparameters and run metadata
            )

    def _on_step(self) -> bool:
        """Log specific environment attributes to W&B every 50 steps."""
        if self.n_calls % 50 == 0:
            # Retrieve the 'last_dsp_params' attribute from the vectorized environment
            params = self.training_env.get_attr('last_dsp_params')[0]
            wandb.log({
                "dsp/jitter": params['jitter'],
                "dsp/bitrate": params['bitrate'],
                "dsp/tilt": params['tilt'],
                "global_step": self.num_timesteps
            })
        return True
    