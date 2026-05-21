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

def compute_attack_reward(score: float, self: object) -> tuple:
    """
    Standardized reward and termination logic for KON-Artist.
    Ensures consistency between single-env and vectorized modes.

    Args:
        score (float): The probability of the attack being classified as 'Bonafide'.
        self (object): The environment instance (AudioAttackEnv or VecDetectorWrapper) to access thresholds and logging.

    Returns:
        reward (float): The computed reward for the current step.
        terminated (bool): Whether the episode should be terminated based on the attack success.
    """
    # Configuration recovery (with defaults for the Wrapper)
    success_threshold = getattr(self, "success_threshold", 0.5)
    current_step = getattr(self, "current_step", "N/A")
    total_timesteps = getattr(self, "total_timesteps", None)
    last_params = getattr(self, "last_dsp_params", "Batched")
    worker_id = getattr(self, "current_worker", "N/A")
    seed = getattr(self, "current_seed", "N/A")
    
    # PROGRESS FORMATTING
    if isinstance(current_step, int) and total_timesteps:
        progress = (current_step / total_timesteps) * 100
        step_str = f"{current_step}/{total_timesteps} ({progress:.1f}%)"
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

    # TWO-STAGE PROGRESSIVE REWARD ACCELERATION
    # Stage 1: Intermediate Directional Signal (10% milestone)
    if score > 0.10:
        reward += 25.0
        bonus_applied = 25.0
        logger.debug(f"Stage 1 Bonus (+25) applied. Current reward: {reward}")

    # Stage 2: Ultimate Evasion Objective (Definitive bypass threshold)
    if score > 0.50:
        reward += 100.0
        logger.info(f"--- ATTACK SUCCESSFUL (Stage 2): Score {score:.4f} ---")
        bonus_applied = 100.0
        logger.debug(f"Stage 2 Bonus (+100) applied. Current reward: {reward}")

    # COMPLETION CRITERIA
    # If the score exceeds a certain threshold, we can consider the episode successful
    terminated = bool(score > success_threshold) # Logic for completion (Agent successfully spoofed the detector)
        
    # Telemetry Logging
    # Only logs specific step info if running in single-env mode
    logger.info(f"Worker {worker_id} | Step {step_str} | DSP: {last_params} | Score: {score:.4f} | Reward: {reward:.2f}")

    return reward, terminated, bonus_applied
