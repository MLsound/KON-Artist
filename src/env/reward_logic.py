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
    last_params = getattr(self, "last_dsp_params", "Batched")
    
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
    
    # COMPLETION CRITERIA
    # If the score exceeds a certain threshold, we can consider the episode successful
    terminated = bool(score > success_threshold) # Logic for completion (Agent successfully spoofed the detector)
    
    # SUCCESS BONUS
    if terminated:
        # Double the reward if we bypass the model (>0.5)
        # This creates a massive 'gravity' pull toward the bonafide class.
        success_bonus = 50.0 # Large bonus to create a strong incentive for successful attacks
        reward += success_bonus
        logger.info(f"--- ATTACK SUCCESSFUL: Score {score:.4f} ---")
        logger.debug(f"Reward after success bonus: {reward}") # Debugging statement to trace reward after success bonus
        
    # Telemetry Logging
    # Only logs specific step info if running in single-env mode
    logger.info(f"Step {current_step} | DSP: {last_params} | Score: {score:.4f} | Reward: {reward:.2f}")

    return reward, terminated
