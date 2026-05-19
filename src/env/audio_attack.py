"""
Gymnasium Environment for Audio Spoofing Attack Simulations.

This module implements an RL environment where an agent attempts to modify 
synthetic or spoofed audio signals using a DSP pipeline. The objective is 
to maximize the probability of the audio being classified as 'Bonafide' 
by a target AASIST3 detector.
"""
import gymnasium as gym
import numpy as np
import itertools
import torch
from src.env.reward_logic import compute_attack_reward
from src.synthesis.dsp import DSPPipeline
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.INFO)  # Set to DEBUG for detailed trace during environment interactions

class AudioAttackEnv(gym.Env):
    """
    RL Environment for optimizing audio spoofing parameters.
    Supports both direct inference and vectorized batch inference modes.
    """
    def __init__(self, detector=None, dsp_config: dict = None, audio_config: dict = None, audio_files=None):
        super().__init__()
        self.detector = detector
        self.audio_config = audio_config
        
        # Extract rank and seed for identifying logs per worker
        self.rank = audio_config.get("rank", 0) if audio_config else 0
        self.seed = audio_config.get("seed", 42) if audio_config else 42
        
        # Initialize audio_files iterator
        if audio_files is not None:
            self.audio_files = itertools.cycle(audio_files) # Create an infinite iterator from the audio stream generator
        else:
            self.audio_files = None # Will be lazily initialized in reset() if audio_config is provided
        
        # Instantiate the DSP pipeline as a persistent object
        self.dsp_config = dsp_config or {}
        self.dsp = DSPPipeline(sample_rate=16000)
        
        # 7-dimensional action space (Jitter, Shimmer, Tilt, etc.)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        # Observation space: 160-dim embedding (if detector present) or 64600-dim raw audio
        if self.detector:
            self.observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(160,), dtype=np.float32)
        else:
            # Fixed AASIST3 length: 64600 samples
            self.observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(64600,), dtype=np.float32)
        
        self.current_audio = None
        self.last_dsp_params = None
        
        # Initialize internal step tracking
        self.current_step = 0
        self.step_limit = 10 # Max steps per episode to prevent infinite loops
        self.success_threshold = 0.5  # 50% probability of Bonafide

    def _denormalize(self, action: np.ndarray) -> dict:
        """
        Maps normalized RL actions [-1, 1] to actual DSP parameter ranges.
        Crucial for the system's monitoring of crop-like audio artifacts.
        """
        # Linear mapping based on your dsp_config ranges
        return {
            'jitter': action[0],
            'shimmer': action[1],
            'tilt': action[2],
            'harmonics': action[3],
            'threshold': action[4] * 20 - 40,  # Example: map [-1, 1] to [-60, -20] dB
            'ratio': (action[5] + 1) * 5 + 1,   # Example: map [-1, 1] to [1, 11] ratio
            'bitrate': int((action[6] + 1) * 64000 + 32000) # Map to bps
        }
        
    def _get_obs(self, embedding):
        """Ensures observations are 1D float32 arrays for PPO buffer compatibility."""
        # Flatten (1, 160) -> (160,) and ensure float32
        if isinstance(embedding, torch.Tensor):
            return embedding.cpu().numpy().flatten().astype(np.float32)
        return embedding.flatten().astype(np.float32)

    def step(self, action: np.ndarray):
        """
        Executes one attack step.
        """
        # Initialize flags for episode termination and truncation
        terminated = False
        truncated = False

        # Increment the internal step counter for episode management
        self.current_step += 1
        
        # 1. Apply DSP transformations
        logger.debug(f"Action received: {action}") # Debugging statement to trace actions
        self.last_dsp_params = self._denormalize(action)
        logger.debug(f"Denormalized DSP parameters: {self.last_dsp_params}") # Debugging statement to trace DSP parameters

        # Process the current audio with the DSP pipeline using the denormalized parameters
        processed_audio = self.dsp(self.current_audio, self.last_dsp_params)
        logger.debug(f"Processed audio shape: {processed_audio.shape}") # Debugging statement to trace audio processing
        
        # Check for truncation (step limit) regardless of mode
        truncated = bool(self.current_step >= self.step_limit)
        if truncated:
            logger.debug(f"Step limit {self.step_limit} reached. Truncating episode.")

        # If no detector is provided, return raw audio for batch inference in a wrapper
        if self.detector is None:
            obs = self._get_obs(processed_audio)
            # Return dummy values for reward and termination; to be filled by VecEnvWrapper
            return obs, 0.0, False, truncated, {
                'processed_audio': processed_audio, 
                'dsp_params': self.last_dsp_params,
                'worker_id': self.rank,
                'seed': self.seed,
                'truncated': truncated
            }

        # 2. Evaluation by AASIST3
        # Capture both score and embedding for reward and observation
        scores, embeddings = self.detector.get_score_and_embedding(processed_audio)
        score, embedding = scores[0], embeddings[0] # Assuming batch size of 1 for direct inference mode
        logger.debug(f"Embedding shape: {embedding.shape}") # Debugging statement to trace embedding extraction
        logger.debug(f"AASIST3 score: {score}") # Debugging statement to trace model output
        obs = self._get_obs(embedding) # Observation: The embedding from AASIST3 (160-dim)
                
        # 3. Compute reward and check for termination
        reward, terminated = compute_attack_reward(score, self) # Centralized call ensures logic parity with non-vectorized env

        # Collect info for logging and analysis
        info = {
            'score': float(score),
            'reward': reward,
            'dsp_params': self.last_dsp_params,
            'terminated': terminated,
            'truncated': truncated,
            'worker_id': self.rank,
            'seed': self.seed
        }

        return obs, reward, terminated, truncated, info
    
    
    def reset(self, seed=None, options=None):
        """Resets the environment to a new audio sample."""
        super().reset(seed=seed)
        self.current_step = 0 # Reset step counter at the beginning of each episode
        
        # Initial info dict for reset
        info = {'worker_id': self.rank, 'seed': self.seed}
        
        # Lazy initialization of the data stream (critical for SubprocVecEnv)
        if self.audio_files is None and self.audio_config is not None:
            from src.data.loader import get_asvspoof_loader, generator_from_ds
            # Filter config to only include arguments expected by get_asvspoof_loader
            loader_config = {k: v for k, v in self.audio_config.items() if k != 'rank'}
            logger.info(f"Initializing worker audio stream with config: {loader_config}")
            # Fetch the next preprocessed audio tensor from the stream
            ds = get_asvspoof_loader(**loader_config)
            self.audio_files = itertools.cycle(generator_from_ds(ds)) # Infinite cycling generator
        
        if self.audio_files is None:
            raise RuntimeError("Environment has no audio source. Provide audio_files or audio_config.")

        try:
            self.current_audio, target_label = next(self.audio_files)
            logger.info(f"Environment Reset: Loading new sample (Label: {'Bonafide' if target_label == 1 else 'Spoof'})")
        except StopIteration:
            # Fallback in case the streaming dataset exhausts
            logger.error("Audio stream exhausted.")
            raise RuntimeError("Audio stream exhausted.")
        
        if self.detector is None:
            # Return raw audio for initial observation
            return self._get_obs(self.current_audio), info

        # Initial inference
        _, embeddings = self.detector.get_score_and_embedding(self.current_audio)
        obs = self._get_obs(embeddings[0]) # Observation: The embedding from AASIST3 (160-dim)

        return obs, info
    