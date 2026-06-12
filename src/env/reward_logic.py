"""
This module defines the reward logic for the KON-Artist environment,
ensuring consistency between single-environment and vectorized modes.

The reward is based on the log-probability of the attack being classified
as 'Bonafide' by the target model, with additional bonuses for successful attacks.

The module also includes termination criteria based on the attack score and step limits.
"""
import numpy as np
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.INFO)  # Set to DEBUG for detailed trace during environment interactions

def compute_attack_reward(score: float, self: object, dsp_params: dict = None) -> tuple:
    """
    Standardized reward and termination logic for KON-Artist.
    Ensures consistency between single-env and vectorized modes.

    Args:
        score (float): The probability of the attack being classified as 'Bonafide'.
        self (object): The environment instance (AudioAttackEnv or VecDetectorWrapper) to access thresholds and logging.
        dsp_params (dict, optional): The DSP configurations used in the current step.

    Returns:
        reward (float): The computed reward for the current step.
        terminated (bool): Whether the episode should be terminated based on the attack success.
    """
    # Configuration recovery (with defaults for the Wrapper)
    success_threshold = getattr(self, "success_threshold", 0.5)
    current_step = getattr(self, "current_step", "N/A")
    total_timesteps = getattr(self, "total_timesteps", None)
    
    # Use provided dsp_params, fallback to environment attribute, then to "Batched"
    last_params = dsp_params or getattr(self, "last_dsp_params", "Batched")
    
    worker_id = getattr(self, "current_worker", "N/A")
    seed = getattr(self, "current_seed", "N/A")
    
    # PROGRESS FORMATTING
    initial_checkpoint_steps = getattr(self, "initial_checkpoint_steps", 0)
    session_total_steps = getattr(self, "session_total_steps", None)

    if isinstance(current_step, int):
        relative_step = current_step - initial_checkpoint_steps
        # Fallback to total_timesteps - initial_checkpoint_steps if session_total_steps is not provided
        total_steps_budget = session_total_steps or (total_timesteps - initial_checkpoint_steps if total_timesteps else None)
        
        if total_steps_budget and total_steps_budget > 0:
            relative_progress = (relative_step / total_steps_budget) * 100
            step_str = f"{relative_step}/{total_steps_budget} ({relative_progress:.2f}%)"
        elif total_timesteps:
            # Fallback to cumulative calculation if we don't have relative budget
            progress = (current_step / total_timesteps) * 100
            step_str = f"{current_step}/{total_timesteps} ({progress:.1f}%)"
        else:
            step_str = f"{current_step}"
    else:
        step_str = f"{current_step}"
    
    # REWARD FUNCTION
    # Reward: Log-probability of appearing 'Bonafide'
    # R = log(P + epsilon) magnifies gradients for low-probability states
    epsilon = 1e-9 # Small constant to prevent log(0) and stabilize training when scores are very low
    vertical_shift = 25.0 # Shift to keep rewards positive for PPO stability
    reward = float(np.log(score + epsilon)) # Logarithmic reward to create a strong gradient signal
    logger.debug(f"Raw Score: {float(score)} | Log Reward: {reward}")
    # Optional: Normalize it to a slightly positive/bounded scale for PPO
    reward += vertical_shift # shift it so the minimum expected log (-25) becomes 0
    logger.debug(f"Shifted Log Reward: {reward}") # Debugging statement to trace shifted reward calculation

    # CONTINUOUS REWARD ACCELERATION
    # Scale continuously with the AASIST3 detector's confidence score to provide a smooth gradient signal.
    # The multiplier is controlled by the 'bonus_amount' hyperparameter.
    bonus_enabled = getattr(self, "bonus", True)
    if bonus_enabled:
        bonus_amount = getattr(self, "bonus_amount", 250.0)
        bonus_applied = float(score) * bonus_amount
        reward += bonus_applied
        logger.debug(f"Continuous Bonus (+{bonus_applied:.2f}) applied. Current reward: {reward}")
    else:
        bonus_applied = 0.0
        logger.debug("Continuous Bonus disabled in configuration.")

    # COMPLETION CRITERIA
    # If the score exceeds a certain threshold, we can consider the episode successful
    terminated = bool(score > success_threshold) # Logic for completion (Agent successfully spoofed the detector)
    
    if terminated:
        logger.info(f"--- ATTACK SUCCESSFUL: Score {score:.4f} ---")
        
    # Telemetry Logging
    # Only logs specific step info if running in single-env mode
    logger.info(f"Worker {worker_id} | Step {step_str} | Score: {score:.4f} | Reward: {reward:.2f} | Bonus: {bonus_applied:.2f} | DSP: {last_params} ")

    return reward, terminated, bonus_applied
