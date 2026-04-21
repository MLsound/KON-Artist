"""
Gymnasium Environment for Audio Spoofing Attack Simulations.

This module implements an RL environment where an agent attempts to modify 
synthetic or spoofed audio signals using a DSP pipeline. The objective is 
to maximize the probability of the audio being classified as 'Bonafide' 
by a target AASIST3 detector.
"""
import gymnasium as gym
import numpy as np
from src.synthesis.dsp import DSPPipeline
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.DEBUG)  # Set to DEBUG for detailed trace during environment interactions

class AudioAttackEnv(gym.Env):
    """
    RL Environment for optimizing audio spoofing parameters.
    """
    def __init__(self, detector, dsp_config: dict, audio_files: list):
        super().__init__()
        self.detector = detector
        self.audio_files = audio_files
        self.dsp_config = dsp_config
        
        # Instantiate the DSP pipeline as a persistent object
        self.dsp = DSPPipeline(sample_rate=16000)
        
        # 7-dimensional action space (Jitter, Shimmer, Tilt, etc.)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        # Observation: 160-layer AASIST3 embedding
        self.observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(160,), dtype=np.float32)
        
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
        return embedding.cpu().numpy().flatten().astype(np.float32)

    def step(self, action: np.ndarray):
        """
        Executes one attack step.
        """
        self.current_step += 1
        
        # 1. Apply DSP transformations
        logger.debug(f"Action received: {action}") # Debugging statement to trace actions
        self.last_dsp_params = self._denormalize(action)
        logger.debug(f"Denormalized DSP parameters: {self.last_dsp_params}") # Debugging statement to trace DSP parameters

        # Call the class instance instead of the old function
        processed_audio = self.dsp(self.current_audio, self.last_dsp_params)
        logger.debug(f"Processed audio shape: {processed_audio.shape}") # Debugging statement to trace audio processing
        
        # 2. Evaluation by AASIST3
        score, embedding = self.detector.get_score_and_embedding(processed_audio)
        logger.debug(f"Embedding shape: {embedding.shape}") # Debugging statement to trace embedding extraction
        logger.debug(f"AASIST3 score: {score}") # Debugging statement to trace model output
        obs = self._get_obs(embedding) # Observation: The embedding from AASIST3 (160-dim)

        # 3. Reward: Probability of appearing 'Bonafide'
        # Cast to standard Python float just to be safe
        # reward = float(score) 
        # logger.debug(f"Reward calculated: {reward}") # Debugging statement to trace reward calculation
        
        # 3. Reward: Log-probability of appearing 'Bonafide'
        # R = log(P + epsilon) magnifies gradients for low-probability states
        epsilon = 1e-9
        reward = float(np.log(score + epsilon))
        logger.debug(f"Raw Score: {float(score)} | Log Reward: {reward}")
        # Optional: Normalize it to a slightly positive/bounded scale for PPO
        reward += 25.0 # e.g., shift it so the minimum expected log (-20) becomes 0

        # Use the internal counter for verbosity
        logger.info(f"Step {self.current_step} | DSP Params: {self.last_dsp_params} | Reward: {float(score)}")
        
        # COMPLETION CRITERIA
        # If the score exceeds a certain threshold, we can consider the episode successful
        # Logic for completion (Agent successfully spoofed the detector)
        terminated = bool(score > self.success_threshold)
        # Logic for step limit (Environment-enforced limit)
        truncated = bool(self.current_step >= self.step_limit)
        
        # Clean output instead of crashing
        if truncated:
            logger.debug(f"Step limit {self.step_limit} reached. Truncating episode.")

        # Collect info for logging and analysis
        info = {
            'score': float(score),
            'reward': reward,
            'dsp_params': self.last_dsp_params,
            'terminated': terminated,
            'truncated': truncated
        }

        return obs, reward, terminated, truncated, info
    
    
    def reset(self, seed=None, options=None):
        """Resets the environment to a new audio sample."""
        super().reset(seed=seed)
        self.current_step = 0 # Reset step counter at the beginning of each episode
        
        try:
            # self.audio_files is now a generator yielding (tensor, label)
            # Fetch the next preprocessed audio tensor from the stream
            self.current_audio, target_label = next(self.audio_files)
            logger.info(f"Environment Reset: Loading new sample (Label: {'Bonafide' if target_label == 1 else 'Spoof'})")
        except StopIteration:
            # Fallback in case the streaming dataset exhausts
            logger.error("Intel system: Audio stream exhausted.")
            raise RuntimeError("Audio stream exhausted. Consider recreating the generator or looping the dataset.")
        
        # The tensor is already preprocessed by loader.py, pass it directly
        _, embedding = self.detector.get_score_and_embedding(self.current_audio)
        obs = self._get_obs(embedding) # Observation: The embedding from AASIST3 (160-dim)

        return obs, {}
    