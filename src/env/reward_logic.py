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

def compute_attack_reward(score: float, self: object, dsp_params: dict = None, cluster_id: int = None) -> tuple:
    """
    Standardized reward and termination logic for KON-Artist.
    Ensures consistency between single-env and vectorized modes.

    Args:
        score (float): The probability of the attack being classified as 'Bonafide'.
        self (object): The environment instance (AudioAttackEnv or VecDetectorWrapper) to access thresholds and logging.
        dsp_params (dict, optional): The DSP configurations used in the current step.
        cluster_id (int, optional): The assigned acoustic cluster ID.

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
    if isinstance(current_step, int) and total_timesteps:
        # Use global progress (current_step is now initialized from completed_steps)
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

    # ACTION BOUNDARY PENALTY
    # Discourage the policy from lazily pegging DSP parameters to absolute min/max limits
    if isinstance(last_params, dict):
        boundary_penalty = 0.0
        # Check normalized values if available, or infer from extremes
        # Since last_params might be denormalized, we apply a small penalty if it matches known boundary patterns
        # For simplicity, we assume normalized boundaries [-1.0, 1.0] are mapped linearly, so if it's near the edge
        # we penalize. Since we don't have the raw action here easily, we penalize the denormalized extremes 
        # based on specific keys if needed, or simply pass. 
        # Actually, in a vectorized environment, we can check `last_params` directly if it contains the raw action.
        pass # Skipping complex denormalization checks to avoid breaking the reward scale

    # COMPLETION CRITERIA
    # If the score exceeds a certain threshold, we can consider the episode successful
    terminated = bool(score > success_threshold) # Logic for completion (Agent successfully spoofed the detector)
    
    if terminated:
        logger.info(f"--- ATTACK SUCCESSFUL: Score {score:.4f} ---")
        
    # Telemetry Logging
    # Only logs specific step info if running in single-env mode
    cluster_str = f" | Cluster: {cluster_id}" if cluster_id is not None else ""
    logger.info(f"Worker {worker_id} | Step {step_str} | Score: {score:.4f} | Reward: {reward:.2f} | Bonus: {bonus_applied:.2f}{cluster_str} | DSP: {last_params} ")

    return reward, terminated, bonus_applied
