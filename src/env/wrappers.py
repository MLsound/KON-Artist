"""
Vectorized Wrappers for KON-Artist Environments.

This module provides wrappers for Stable-Baselines3 vectorized environments,
enabling batched inference for the AASIST3 detector. This decouples
expensive GPU-based model execution from CPU-based DSP transformations,
maximizing throughput during RL training.
"""
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnvWrapper
from src.env.reward_logic import compute_attack_reward

class VecDetectorWrapper(VecEnvWrapper):
    """
    VecEnvWrapper that performs batched detector inference on observations.
    
    Expects the underlying environment to return raw audio waveforms as observations.
    Transforms these waveforms into AASIST3 embeddings for the RL agent.
    """
    def __init__(self, venv, detector, bonus: bool = True, bonus_amount: float = 250.0, config: dict = None, initial_checkpoint_steps: int = 0, session_total_steps: int = None):
        super().__init__(venv)
        self.detector = detector
        self.bonus = bonus
        self.bonus_amount = bonus_amount
        self.config = config or {}
        self.initial_checkpoint_steps = initial_checkpoint_steps
        self.session_total_steps = session_total_steps
        # Override observation space to be the embedding space
        self.observation_space = gym.spaces.Box(
            low=-1e5, high=1e5, shape=(160,), dtype=np.float32
        )
        self.success_threshold = 0.5
        
        # Initialize step counter from completed_steps to maintain global progress on resume
        self.total_steps = self.config.get('ppo', {}).get('completed_steps', 0) 

    def reset(self):
        """Batched reset for all environments."""
        obs = self.venv.reset()
        # obs is [N_ENVS, 64600]
        waveform = torch.from_numpy(obs).unsqueeze(1).float() # [N, 1, L]
        
        _, embeddings = self.detector.get_score_and_embedding(waveform)
        return embeddings.cpu().numpy()

    def step_wait(self):
        """Batched step wait for all environments."""
        obs, rewards, dones, infos = self.venv.step_wait()
        # obs is [N_ENVS, 64600]
        self.total_steps += self.num_envs
        
        # Prepare waveforms for batch inference
        # 1. Current observations (for next step or if it just finished)
        waveforms = [torch.from_numpy(o).unsqueeze(0) for o in obs]
        # 2. Terminal observations (if an environment was reset by SubprocVecEnv)
        terminal_map = {} # Maps env_idx to position in waveforms list for terminal obs
        
        for i, done in enumerate(dones):
            if done and "terminal_observation" in infos[i]:
                term_obs = infos[i]["terminal_observation"]
                terminal_map[i] = len(waveforms)
                waveforms.append(torch.from_numpy(term_obs).unsqueeze(0))
        
        # Combined batch inference
        batch_waveform = torch.cat(waveforms, dim=0).unsqueeze(1).float() # [Total, 1, L]
        scores, embeddings = self.detector.get_score_and_embedding(batch_waveform)
        
        new_rewards = []
        new_dones = []
        
        for i in range(self.num_envs):
            # If the environment finished in the worker, the reward for the STEP 
            # that just finished must come from the 'terminal_observation'.
            if i in terminal_map:
                idx = terminal_map[i]
                score = scores[idx]
                # Replace terminal raw audio with embedding to prevent crash in SB3
                infos[i]["terminal_observation"] = embeddings[idx].cpu().numpy().flatten().astype(np.float32)
            else:
                score = scores[i]
            
            # Extract worker-specific info for logging
            self.current_worker = infos[i].get('worker_id', 'N/A')
            self.current_seed = infos[i].get('seed', 'N/A')

            # Update current_step for logging in reward_logic
            self.current_step = self.total_steps
            # Extract DSP parameters used in this worker's step
            dsp_params = infos[i].get('dsp_params', None)
            
            # Compute reward and check for termination based on the CORRECT score
            reward, terminated, bonus = compute_attack_reward(score, self, dsp_params=dsp_params)
            
            new_rewards.append(reward)
            # SB3 VecEnv handles 'dones' (terminated or truncated)
            # We add our own 'terminated' condition from the detector
            new_dones.append(dones[i] or terminated)
            
            # Update info dictionaries with actual detector results
            infos[i]['score'] = float(score)
            infos[i]['reward'] = reward
            infos[i]['bonus'] = bonus
            infos[i]['terminated'] = terminated
            
        # Return embeddings of current observations [0:num_envs]
        return embeddings[:self.num_envs].cpu().numpy(), np.array(new_rewards), np.array(new_dones), infos


import threading
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.INFO)

class AdaptiveActionSpaceClipsWrapper(gym.Wrapper):
    """
    Gymnasium environment wrapper that dynamically compresses the continuous action space
    exploration boundaries based on a global rolling Attack Success Rate (ASR) threshold.
    Anchors compression toward a global moving average of successful actions.
    """
    def __init__(self, env: gym.Env, config: dict):
        super().__init__(env)
        self.config = config or {}
        self.success_threshold = getattr(env, "success_threshold", 0.50)
        
        # Load parameters from configuration
        clip_cfg = self.config.get("clipping", {})
        self.asr_compression_threshold = float(clip_cfg.get("asr_compression_threshold", 0.35))
        self.max_compression_factor = float(clip_cfg.get("max_compression_factor", 0.80))
        self.compression_alpha = float(clip_cfg.get("compression_alpha", 2.0))
        self.asr_window_size = int(clip_cfg.get("asr_update_window", 4096))
        
        # Thread-safe rolling window arrays
        self._lock = threading.Lock()
        self.asr_history = np.zeros(self.asr_window_size, dtype=np.float32)
        self.asr_count = 0
        self.asr_pointer = 0
        
        # Buffer to track successful actions (for computing global moving average anchor)
        action_dim = env.action_space.shape[0]
        self.successful_actions_buffer = np.zeros((self.asr_window_size, action_dim), dtype=np.float32)
        self.successful_actions_count = 0
        self.successful_actions_pointer = 0

    def update_trackers(self, score: float, action: np.ndarray):
        """Updates the rolling window trackers for ASR and successful actions."""
        with self._lock:
            is_success = 1.0 if score >= self.success_threshold else 0.0
            
            # Update rolling ASR
            self.asr_history[self.asr_pointer] = is_success
            self.asr_pointer = (self.asr_pointer + 1) % self.asr_window_size
            self.asr_count = min(self.asr_count + 1, self.asr_window_size)
            
            # If successful, track the action that led to it
            if is_success == 1.0:
                self.successful_actions_buffer[self.successful_actions_pointer] = action
                self.successful_actions_pointer = (self.successful_actions_pointer + 1) % self.asr_window_size
                self.successful_actions_count = min(self.successful_actions_count + 1, self.asr_window_size)

    def get_asr(self) -> float:
        """Computes the global rolling Attack Success Rate (ASR)."""
        with self._lock:
            if self.asr_count == 0:
                return 0.0
            return float(np.sum(self.asr_history[:self.asr_count]) / self.asr_count)

    def get_successful_actions_average(self) -> np.ndarray:
        """Computes the global moving average of successful actions."""
        with self._lock:
            if self.successful_actions_count == 0:
                # Default anchor is the center of the normalized action space (all zeros)
                return np.zeros(self.action_space.shape, dtype=np.float32)
            return np.mean(self.successful_actions_buffer[:self.successful_actions_count], axis=0)

    def step(self, action: np.ndarray):
        """Intercepts action, applies global contraction, and executes environment step."""
        anchor = self.get_successful_actions_average()
        asr = self.get_asr()
        
        # Calculate compression factor lambda based on global ASR
        if asr <= self.asr_compression_threshold:
            lambda_val = 1.0
        else:
            delta = (asr - self.asr_compression_threshold) / (1.0 - self.asr_compression_threshold)
            delta = np.clip(delta, 0.0, 1.0)
            lambda_val = 1.0 - (self.max_compression_factor * (delta ** self.compression_alpha))
        
        # Contract action space limits and compute compressed action
        action_compressed = anchor + lambda_val * (action - anchor)
        action_compressed = np.clip(action_compressed, self.env.action_space.low, self.env.action_space.high)
        
        # Compute dynamic bounds relative to action space low/high
        min_allowed_action = anchor + lambda_val * (self.env.action_space.low - anchor)
        max_action = anchor + lambda_val * (self.env.action_space.high - anchor)
        
        # Log the global contraction metrics
        logger.info(
            "Global ASR: %.4f | Compression Scalar: %.4f | Min Bound: %s | Max Bound: %s",
            asr, lambda_val, str(min_allowed_action), str(max_action)
        )
        
        # Execute environment step using the compressed action
        obs, reward, terminated, truncated, info = self.env.step(action_compressed)
        
        # Update rolling windows based on step outcome
        if "score" in info:
            self.update_trackers(info["score"], action_compressed)
            
        # Expose contraction boundary telemetry in info dictionary
        info["min_allowed_action"] = min_allowed_action
        info["max_action"] = max_action
        info["compression_scalar"] = lambda_val
        
        return obs, reward, terminated, truncated, info
