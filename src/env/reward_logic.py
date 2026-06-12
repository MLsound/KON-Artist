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

def compute_attack_reward(score: float, self: object, dsp_params: dict = None, cluster_probs: np.ndarray = None) -> tuple:
    """
    Standardized reward and termination logic for KON-Artist.
    Ensures consistency between single-env and vectorized modes.

    Args:
        score (float): The probability of the attack being classified as 'Bonafide'.
        self (object): The environment instance (AudioAttackEnv or VecDetectorWrapper) to access thresholds and logging.
        dsp_params (dict, optional): The DSP configurations used in the current step.
        cluster_probs (np.ndarray, optional): Soft cluster assignment probabilities.

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
    base_log_reward = float(np.log(score + epsilon)) # Logarithmic reward to create a strong gradient signal
    logger.debug(f"Raw Score: {float(score)} | Log Reward: {base_log_reward}")
    
    # Base reward with vertical shift applied (Unscaled Axis)
    base_reward = base_log_reward + vertical_shift

    # A) CLUSTER MULTIPLIER EXTRACTION
    cluster_id = None
    cluster_multiplier = 1.0
    clustering_cfg = getattr(self, "clustering_config", {}) or {}
    if cluster_probs is not None:
        cluster_id = int(np.argmax(cluster_probs))
        multipliers_dict = clustering_cfg.get("inverse_frequency_multipliers", {}) or {}
        if multipliers_dict:
            # Convert dict to array matched to cluster_probs length
            n_clusters = len(cluster_probs)
            multipliers_arr = np.array([multipliers_dict.get(i, 1.0) for i in range(n_clusters)], dtype=np.float32)
            # Compute soft-assignment probability vector dot product
            cluster_multiplier = float(np.dot(cluster_probs, multipliers_arr))
            logger.debug(f"Cluster-Aware Multiplier computed: {cluster_multiplier:.2f} (Cluster {cluster_id})")

    # B) STAGNATION PENALTY & FIDELITY SHAPING (Unscaled Axis)
    stagnation_penalty = 0.0
    fidelity_penalty = 0.0
    aux_cfg = clustering_cfg.get("auxiliary_reward", {}) or {}
    if aux_cfg.get("enabled", False):
        # 1. Stagnation Penalty: if score is exactly 0.0, the agent is in a dead zone
        if float(score) <= 1e-7: # Practical zero for AASIST3
            stagnation_penalty = float(aux_cfg.get("stagnation_penalty", -5.0))
            logger.debug(f"Stagnation Penalty ({stagnation_penalty}) applied for zero-score.")

        # 2. Fidelity Guidance: penalize extreme acoustic damage
        if isinstance(last_params, dict):
            # Targets: ratio=1.0, bitrate=160000
            target_ratio = 1.0
            target_bitrate = 160000.0
            
            curr_ratio = float(last_params.get('ratio', 1.0))
            curr_bitrate = float(last_params.get('bitrate', 160000.0))
            
            # Normalize distance (Ratio max ~11, Bitrate min ~32000)
            dist_ratio = (curr_ratio - target_ratio) / 10.0
            dist_bitrate = (target_bitrate - curr_bitrate) / 128000.0
            
            l2_fidelity_loss = np.sqrt(dist_ratio**2 + dist_bitrate**2)
            fidelity_weight = aux_cfg.get("fidelity_weight", 2.0)
            fidelity_penalty = float(-l2_fidelity_loss * fidelity_weight)
            logger.debug(f"Fidelity Penalty ({fidelity_penalty:.2f}) applied. L2 Dist: {l2_fidelity_loss:.4f}")

    # C) DECOUPLED AND SCALED SUCCESS BONUS
    bonus_applied = 0.0
    bonus_enabled = getattr(self, "bonus", True)
    if bonus_enabled and float(score) > success_threshold:
        bonus_amount = getattr(self, "bonus_amount", 250.0)
        # Calculate proportional bonus scaled strictly by cluster multiplier
        raw_bonus = float(score) * bonus_amount
        bonus_applied = cluster_multiplier * raw_bonus
        logger.debug(f"Success Bonus (+{bonus_applied:.2f}) applied. Raw: {raw_bonus:.2f}, Mult: {cluster_multiplier:.2f}")

    # Unified reward assembly (preserving unscaled baseline penalties)
    reward = base_reward + stagnation_penalty + fidelity_penalty + bonus_applied

    # COMPLETION CRITERIA
    terminated = bool(score > success_threshold) 
    
    if terminated:
        logger.info(f"--- ATTACK SUCCESSFUL: Score {score:.4f} ---")
        
    # Telemetry Logging
    cluster_str = f" | Cluster: {cluster_id}" if cluster_id is not None else ""
    logger.info(f"Worker {worker_id} | Step {step_str} | Score: {score:.4f} | Reward: {reward:.2f} | Bonus: {bonus_applied:.2f}{cluster_str} | DSP: {last_params} ")

    return np.float32(reward), terminated, float(bonus_applied)
