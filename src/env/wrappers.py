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
    def __init__(self, venv, detector):
        super().__init__(venv)
        self.detector = detector
        # Override observation space to be the embedding space
        self.observation_space = gym.spaces.Box(
            low=-1e5, high=1e5, shape=(160,), dtype=np.float32
        )
        self.success_threshold = 0.5
        self.total_steps = 0 # Global step counter for vectorized environments

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
        
        waveform = torch.from_numpy(obs).unsqueeze(1).float()
        scores, embeddings = self.detector.get_score_and_embedding(waveform)
        
        new_rewards = []
        new_dones = []
        
        for i in range(self.num_envs):
            score = scores[i] # Probability of being classified as 'Bonafide'
            truncated = infos[i].get('truncated', False) # Check if the episode was truncated by the environment
            
            # Extract worker-specific info
            self.current_worker = infos[i].get('worker_id', 'N/A')
            self.current_seed = infos[i].get('seed', 'N/A')

            # Update current_step for logging in reward_logic
            self.current_step = self.total_steps
            # Compute reward and check for termination
            reward, terminated = compute_attack_reward(score, self)
            
            new_rewards.append(reward)
            # SB3 VecEnv handles 'dones' (terminated or truncated)
            new_dones.append(terminated or truncated)
            
            # Update info dictionaries with actual detector results
            infos[i]['score'] = float(score)
            infos[i]['reward'] = reward
            infos[i]['terminated'] = terminated
            
        return embeddings.cpu().numpy(), np.array(new_rewards), np.array(new_dones), infos
