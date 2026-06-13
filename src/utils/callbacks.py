"""
Training Callbacks for Reinforcement Learning (KON-Artist Project).

This module provides custom Stable Baselines3 (SB3) callbacks to monitor 
and log training progress. It includes utilities for local CSV reward 
logging—necessary for convergence plotting—and Weights & Biases (W&B) 
integration for tracking real-time DSP parameter evolution.
"""
import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.INFO)

class RewardLoggerCallback(BaseCallback):
    """
    Callback to save episodic rewards into a CSV file.
    Essential for generating convergence plots for final reports.
    """
    def __init__(self, check_freq: int, id: str, log_dir: str, verbose: int = 1, initial_checkpoint_steps: int = 0, session_total_steps: int = None):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.id = id
        self.initial_checkpoint_steps = initial_checkpoint_steps
        self.session_total_steps = session_total_steps
        self.save_path = os.path.join(log_dir, f'history/rewards_{id}.csv')
        
        # Create directory if it does not exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_training_start(self) -> None:
        """Initialize the CSV file with headers at the start of training."""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'session_step', 'session_progress_pct', 'reward', 'score', 'bonus', 'cluster', 
                'dsp_jitter', 'dsp_shimmer', 'dsp_tilt', 'dsp_harmonics', 'dsp_threshold', 'dsp_ratio', 'dsp_bitrate',
                'raw_action_0', 'raw_action_1', 'raw_action_2', 'raw_action_3', 'raw_action_4', 'raw_action_5', 'raw_action_6',
                'conditioned_action_0', 'conditioned_action_1', 'conditioned_action_2', 'conditioned_action_3', 'conditioned_action_4', 'conditioned_action_5', 'conditioned_action_6'
            ])
        logger.info(f"Reward, Score, Bonus and DSP logging started at: {self.save_path}")

    def _on_step(self) -> bool:
        """
        Log rewards, scores, bonuses and DSP parameters for every step by inspecting environment 'infos'.
        """
        global_step = self.num_timesteps
        session_step = global_step - self.initial_checkpoint_steps
        session_progress_pct = 0.0
        if self.session_total_steps and self.session_total_steps > 0:
            session_progress_pct = (session_step / self.session_total_steps) * 100

        # Record session metrics to TensorBoard logger
        self.logger.record("time/session_step", session_step)
        self.logger.record("time/session_progress_pct", session_progress_pct)

        # Retrieve 'infos' from local variables (available during collect_rollouts)
        infos = self.locals.get("infos")
        if infos is not None:
            # We use 'a' (append) mode to add to the existing file
            with open(self.save_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for info in infos:
                    # 'reward', 'score' and 'bonus' are added to 'info' by our VecDetectorWrapper in every step
                    if all(k in info for k in ["reward", "score", "bonus"]):
                        reward = info["reward"]
                        score = info["score"]
                        bonus = info["bonus"]
                        cluster = info.get("cluster_id", "")
                        dsp = info.get("dsp_params", {})
                        
                        raw_act = info.get("raw_action", np.zeros(7))
                        cond_act = info.get("conditioned_action", np.zeros(7))
                        
                        writer.writerow([
                            global_step, 
                            session_step,
                            session_progress_pct,
                            reward, 
                            score, 
                            bonus,
                            cluster,
                            dsp.get("jitter", ""),
                            dsp.get("shimmer", ""),
                            dsp.get("tilt", ""),
                            dsp.get("harmonics", ""),
                            dsp.get("threshold", ""),
                            dsp.get("ratio", ""),
                            dsp.get("bitrate", ""),
                            raw_act[0], raw_act[1], raw_act[2], raw_act[3], raw_act[4], raw_act[5], raw_act[6],
                            cond_act[0], cond_act[1], cond_act[2], cond_act[3], cond_act[4], cond_act[5], cond_act[6]
                        ])
                        
                    # Also log if an episode finished (optional, but 'reward' above covers the terminal reward too)
                    if self.verbose > 1 and "episode" in info:
                        logger.info(f"Step {global_step}: Episode finished with reward: {info['episode']['r']:.2f}")
            
        return True


class WandbAudioCallback(BaseCallback):
    """
    Callback to log environment-specific DSP parameters to Weights & Biases.
    """
    # 1. Start the W&B session (Required for WandbAudioCallback to function)
    def __init__(self, config: dict, verbose=0, initial_checkpoint_steps: int = 0, session_total_steps: int = None):
        super().__init__(verbose)
        self.initial_checkpoint_steps = initial_checkpoint_steps
        self.session_total_steps = session_total_steps
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
            global_step = self.num_timesteps
            session_step = global_step - self.initial_checkpoint_steps
            session_progress_pct = 0.0
            if self.session_total_steps and self.session_total_steps > 0:
                session_progress_pct = (session_step / self.session_total_steps) * 100

            wandb.log({
                "dsp/jitter": params['jitter'],
                "dsp/bitrate": params['bitrate'],
                "dsp/tilt": params['tilt'],
                "global_step": global_step,
                "session_step": session_step,
                "session_progress_pct": session_progress_pct
            })
        return True

class EntropyDecayCallback(BaseCallback):
    """
    Callback to progressively reduce the entropy coefficient (ent_coef) during training.
    This encourages exploration early on and exploitation later.
    """
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, total_timesteps: int, verbose: int = 0, initial_checkpoint_steps: int = 0, session_total_steps: int = None, mode: str = "global"):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps
        self.initial_checkpoint_steps = initial_checkpoint_steps
        self.session_total_steps = session_total_steps
        self.mode = mode

    def _on_step(self) -> bool:
        # Calculate progress according to the selected mode
        if self.mode == "cycle" and self.session_total_steps is not None and self.session_total_steps > 0:
            current_cycle_step = self.num_timesteps - self.initial_checkpoint_steps
            progress = current_cycle_step / self.session_total_steps
        else:
            progress = self.num_timesteps / self.total_timesteps
            
        progress = max(0.0, min(1.0, progress))
        new_ent_coef = self.initial_ent_coef + (self.final_ent_coef - self.initial_ent_coef) * progress
        
        # Update the ent_coef in the PPO model
        self.model.ent_coef = new_ent_coef
        
        if self.n_calls % 1000 == 0:
            global_step = self.num_timesteps
            session_step = global_step - self.initial_checkpoint_steps
            session_progress_pct = 0.0
            if self.session_total_steps and self.session_total_steps > 0:
                session_progress_pct = (session_step / self.session_total_steps) * 100

            logger.debug(f"Step {global_step}: ent_coef set to {new_ent_coef:.6f}")
            if wandb.run is not None:
                wandb.log({
                    "train/entropy_coefficient": new_ent_coef,
                    "global_step": global_step,
                    "session_step": session_step,
                    "session_progress_pct": session_progress_pct
                })
        
        return True


class LearningRateLoggerCallback(BaseCallback):
    """
    Callback to log the current learning rate to W&B and logs.
    Useful for verifying the annealing schedule.
    """
    def __init__(self, verbose: int = 0, initial_checkpoint_steps: int = 0, session_total_steps: int = None):
        super().__init__(verbose)
        self.initial_checkpoint_steps = initial_checkpoint_steps
        self.session_total_steps = session_total_steps

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            # Retrieve learning rate from the optimizer
            # SB3 stores the current LR in the model.lr_schedule if functional
            # But the actual value used in the optimizer is more definitive
            current_lr = self.model.policy.optimizer.param_groups[0]['lr']
            global_step = self.num_timesteps
            session_step = global_step - self.initial_checkpoint_steps
            session_progress_pct = 0.0
            if self.session_total_steps and self.session_total_steps > 0:
                session_progress_pct = (session_step / self.session_total_steps) * 100
            
            if wandb.run is not None:
                wandb.log({
                    "train/learning_rate": current_lr,
                    "global_step": global_step,
                    "session_step": session_step,
                    "session_progress_pct": session_progress_pct
                })
            
            if self.verbose > 0:
                logger.info(f"Step {global_step}: Current Learning Rate: {current_lr:.2e}")
        return True
    