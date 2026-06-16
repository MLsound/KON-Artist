"""
This module defines the reward logic for the KON-Artist environment,
ensuring consistency between single-environment and vectorized modes.

The reward is based on the log-probability of the attack being classified
as 'Bonafide' by the target model, with additional bonuses for successful attacks.

The module also includes termination criteria based on the attack score and step limits.
"""
import math
import numpy as np
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.INFO)  # Set to DEBUG for detailed trace during environment interactions

def stable_sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def compute_attack_reward(score: float, self: object, dsp_params: dict = None, logit: float = None) -> tuple:
    """
    Standardized piecewise logit-aware reward and termination logic for KON-Artist.
    Ensures consistency between single-env and vectorized modes.

    Args:
        score (float): The probability of the attack being classified as 'Bonafide'.
        self (object): The environment instance (AudioAttackEnv or VecDetectorWrapper) to access thresholds and logging.
        dsp_params (dict, optional): The DSP configurations used in the current step.
        logit (float, optional): Raw un-softmaxed logit from the detector.

    Returns:
        reward (float): The computed reward for the current step.
        terminated (bool): Whether the episode should be terminated based on the attack success.
        bonus (float): The applied success bonus.
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
    
    # RECONSTRUCT LOGIT IF NOT PROVIDED
    if logit is None:
        epsilon_score = 1e-9
        clipped_score = np.clip(score, epsilon_score, 1.0 - epsilon_score)
        logit = float(np.log(clipped_score) - np.log(1.0 - clipped_score))
    
    epsilon = 1e-9
    vertical_shift = 25.0
    
    # PIECEWISE LOGIT-AWARE REWARD
    if logit < 0.0:
        # Stage A: Exploration Phase
        sig = stable_sigmoid(logit)
        base_reward = float(np.log(sig + epsilon)) + vertical_shift
        bonus_applied = 0.0
        logger.debug(f"Stage A - Logit: {logit} | Sigmoid: {sig} | Base Reward: {base_reward}")
    else:
        # Stage B: Exploitation Phase
        base_reward = float(np.log(score + epsilon)) + vertical_shift
        
        # Capped success bonus
        bonus_applied = 0.0
        bonus_enabled = getattr(self, "bonus", True)
        if bonus_enabled and float(score) > success_threshold:
            bonus_amount = getattr(self, "bonus_amount", 250.0)
            scaled_multiplier = math.tanh(float(score) * 2.0)
            bonus_applied = min(bonus_amount * scaled_multiplier, bonus_amount)
            logger.debug(f"Stage B - Success Bonus (+{bonus_applied:.2f}) applied. Reward: {base_reward + bonus_applied}")
            
    reward = base_reward + bonus_applied

    # COMPLETION CRITERIA
    terminated = bool(score > success_threshold)
    
    if terminated:
        logger.info(f"--- ATTACK SUCCESSFUL: Score {score:.4f} ---")
        
    # Telemetry Logging (strictly context-blind, no GMM/clusters)
    logger.info(f"Worker {worker_id} | Step {step_str} | Score: {score:.4f} | Reward: {reward:.2f} | Bonus: {bonus_applied:.2f} | DSP: {last_params}")

    return float(reward), terminated, float(bonus_applied)
