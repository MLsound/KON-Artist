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
    def __init__(self, detector=None, dsp_config: dict = None, audio_config: dict = None, audio_files=None, bonus: bool = True, bonus_amount: float = 250.0, completed_steps: int = 0, clustering_config: dict = None, initial_checkpoint_steps: int = 0, session_total_steps: int = None):
        super().__init__()
        self.detector = detector
        self.audio_config = audio_config
        self.clustering_config = clustering_config
        self.bonus = bonus
        self.bonus_amount = bonus_amount
        self.completed_steps = completed_steps
        self.initial_checkpoint_steps = initial_checkpoint_steps
        self.session_total_steps = session_total_steps
        
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
        
        # Observation space: 160-dim embedding (+ K clustering dims if enabled) or 64600-dim raw audio
        if self.detector:
            obs_dim = 160
            if self.clustering_config:
                # Try to get actual n_components from model file
                import pickle
                import warnings
                try:
                    with open(self.clustering_config["model_path"], "rb") as f:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UserWarning)
                            pipe = pickle.load(f)
                            obs_dim += pipe.named_steps['gmm'].n_components
                except (FileNotFoundError, KeyError):
                    # Fallback to 4 clusters if model is missing
                    obs_dim += 4
            self.observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(obs_dim,), dtype=np.float32)
        else:
            # Fixed AASIST3 length: 64600 samples
            # Note: VecDetectorWrapper will override this to (160 + K,)
            self.observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(64600,), dtype=np.float32)
        
        self.current_audio = None
        self.last_dsp_params = None
        
        # Initialize internal step tracking
        self.current_step = self.completed_steps # Start from completed steps to maintain global progress
        self.step_limit = 10 # Max steps per episode to prevent infinite loops
        self.success_threshold = 0.5  # 50% probability of Bonafide

    def _get_conditioned_observation(self, baseline_embedding):
        """
        Lazily loads the serialized GMM pipeline inside worker processes to avoid pickling failures
        and appends the soft cluster distribution vector to the environment observation.
        """
        if not hasattr(self, "clustering_pipeline"):
            import pickle
            import warnings
            try:
                with open(self.clustering_config["model_path"], "rb") as f:
                    # Suppress sklearn version mismatch warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        self.clustering_pipeline = pickle.load(f)
            except (FileNotFoundError, KeyError) as e:
                logger.error(f"Failed to load clustering pipeline: {e}")
                raise RuntimeError(f"Clustering enabled but pipeline not found at {self.clustering_config.get('model_path')}")
                
        # Ensure standard numpy array formatting [1, 160]
        if isinstance(baseline_embedding, torch.Tensor):
            embedding_np = baseline_embedding.detach().cpu().numpy().reshape(1, -1)
        else:
            embedding_np = np.array(baseline_embedding).reshape(1, -1)
            
        cluster_probabilities = self.clustering_pipeline.predict_proba(embedding_np).flatten().astype(np.float32)
        
        # Cluster Entropy Sanity Check
        max_p = float(np.max(cluster_probabilities))
        if max_p < 0.2:
            logger.warning(f"WARNING: Vague cluster assignment detected (max prob: {max_p:.2f}). Conditioning may be structurally impossible.")
            
        self.current_cluster_probs = cluster_probabilities # Update current probabilities for biasing
        unified_observation = np.concatenate([embedding_np.flatten(), cluster_probabilities])
        
        return unified_observation.astype(np.float32)

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
        """Ensures observations are 1D float32 arrays and applies conditioning if enabled."""
        if self.clustering_config and self.detector is not None:
            return self._get_conditioned_observation(embedding)
            
        # Standard flattening and float32 conversion
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
        
        # B) CLUSTER-CONDITIONED ACTION BIASING
        # Intercept raw action and apply translation matrix
        biased_action = np.copy(action).astype(np.float32)
        cluster_probs = None
        if self.clustering_config and hasattr(self, "current_trajectory_cluster_probs"):
            biases = self.clustering_config.get("cluster_biases")
            if biases is not None:
                # biases shape: [K, Action_Dim]
                bias_matrix = np.array(biases, dtype=np.float32)
                # Compute dynamic bias: dot product of probs and matrix
                dynamic_bias = np.dot(self.current_trajectory_cluster_probs, bias_matrix)
                biased_action = np.clip(biased_action + dynamic_bias, -1.0, 1.0).astype(np.float32)
                logger.debug(f"Action Biased: {action} -> {biased_action} (Bias: {dynamic_bias})")

        # 1. Apply DSP transformations
        logger.debug(f"Action received: {biased_action}") # Debugging statement to trace actions
        self.last_dsp_params = self._denormalize(biased_action)
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
        # Capture both score, logit, and embedding for reward and observation
        try:
            detector_res = self.detector.get_score_and_embedding(processed_audio, return_logits=True)
        except TypeError:
            detector_res = self.detector.get_score_and_embedding(processed_audio)
            
        if len(detector_res) == 3:
            scores, logits, embeddings = detector_res
            score, logit, embedding = scores[0], logits[0], embeddings[0]
        else:
            scores, embeddings = detector_res
            score, embedding = scores[0], embeddings[0]
            # Fallback reconstruction of logit
            epsilon_score = 1e-9
            clipped_score = np.clip(score, epsilon_score, 1.0 - epsilon_score)
            logit = float(np.log(clipped_score) - np.log(1.0 - clipped_score))
            
        logger.debug(f"Embedding shape: {embedding.shape}") # Debugging statement to trace embedding extraction
        logger.debug(f"AASIST3 score: {score}") # Debugging statement to trace model output
        
        if self.clustering_config:
            emb_np = embedding.cpu().numpy().flatten().astype(np.float32)
            obs = np.concatenate([emb_np, self.current_trajectory_cluster_probs]).astype(np.float32)
            cluster_probs = self.current_trajectory_cluster_probs
        else:
            obs = self._get_obs(embedding) # Observation: The embedding from AASIST3 (160-dim)
            cluster_probs = None
                
        # 3. Compute reward and check for termination
        reward, terminated, bonus = compute_attack_reward(score, self, cluster_probs=cluster_probs, logit=logit) # Centralized call ensures logic parity with non-vectorized env

        # Collect info for logging and analysis
        info = {
            'score': float(score),
            'reward': float(reward),
            'bonus': int(bonus),
            'dsp_params': self.last_dsp_params,
            'terminated': terminated,
            'truncated': truncated,
            'worker_id': self.rank,
            'seed': self.seed
        }
        if cluster_probs is not None:
            info['cluster_id'] = int(np.argmax(cluster_probs))

        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    
    def reset(self, seed=None, options=None):
        """Resets the environment to a new audio sample."""
        super().reset(seed=seed)
        self.current_step = 0 # Reset step counter at the beginning of each episode
        self.last_logit = None # Reset stagnation tracking
        
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
            
            # Log the sample label for diagnostic transparency
            logger.info(f"Environment Reset: Loading new sample (Label: {'Bonafide' if target_label == 1 else 'Spoof'})")
            
            # SECOND LAYER OF DEFENSE: Assert strict categorical exclusion
            if target_label == 1: # 1 is Bonafide
                critical_error = "CRITICAL SECURITY BREACH: Bonafide signal detected in attack pipeline. Aborting to protect human voice integrity."
                logger.error(critical_error)
                raise RuntimeError(critical_error)
                
        except StopIteration:
            # Fallback in case the streaming dataset exhausts
            logger.error("Audio stream exhausted.")
            raise RuntimeError("Audio stream exhausted.")
        
        if self.detector is None:
            # Return raw audio for initial observation
            info.update({'label': target_label})
            return self._get_obs(self.current_audio), info

        # Initial inference
        scores, embeddings = self.detector.get_score_and_embedding(self.current_audio)
        score = scores[0]
        
        if self.clustering_config:
            obs = self._get_conditioned_observation(embeddings[0])
            self.current_trajectory_cluster_probs = obs[160:]
            
            # Enforce hard exclusion check recursively
            exclude_clusters = self.clustering_config.get("exclude_clusters", [])
            k_dom = int(np.argmax(self.current_trajectory_cluster_probs))
            
            retries = 0
            max_retries = 100
            while k_dom in exclude_clusters and retries < max_retries:
                retries += 1
                logger.info(f"AudioAttackEnv Reset: Sample belonged to excluded cluster {k_dom}. Discarding and loading next sample (Retry {retries}/{max_retries}).")
                try:
                    self.current_audio, target_label = next(self.audio_files)
                    if target_label == 1:
                        raise RuntimeError("CRITICAL SECURITY BREACH: Bonafide signal detected in attack pipeline.")
                except StopIteration:
                    logger.error("Audio stream exhausted.")
                    raise RuntimeError("Audio stream exhausted.")
                    
                scores, embeddings = self.detector.get_score_and_embedding(self.current_audio)
                score = scores[0]
                obs = self._get_conditioned_observation(embeddings[0])
                self.current_trajectory_cluster_probs = obs[160:]
                k_dom = int(np.argmax(self.current_trajectory_cluster_probs))
                
            if retries >= max_retries:
                logger.warning(f"AudioAttackEnv Reset: Reached max retries ({max_retries}) trying to filter excluded clusters.")
                
            cluster_id = int(np.argmax(self.current_trajectory_cluster_probs))
            info['cluster_id'] = cluster_id
        else:
            obs = self._get_obs(embeddings[0]) # Observation: The embedding from AASIST3 (160-dim)

        info.update({'score': float(score), 'label': target_label})
        return obs.astype(np.float32), info
    