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
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__, level=logging.INFO)

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
        
        # Acoustic Clustering Configuration
        self.clustering_config = self.config.get('clustering', None)
        obs_dim = 160
        if self.clustering_config:
            # Eagerly load the pipeline to determine the actual number of clusters
            import pickle
            import warnings
            try:
                with open(self.clustering_config["model_path"], "rb") as f:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        self.clustering_pipeline = pickle.load(f)
                
                # Determine n_clusters from the GMM step in the pipeline
                n_clusters = self.clustering_pipeline.named_steps['gmm'].n_components
                obs_dim += n_clusters
                logger.info(f"Clustering enabled for VecDetectorWrapper. Clusters detected: {n_clusters}")
            except (FileNotFoundError, KeyError) as e:
                logger.error(f"Failed to load clustering pipeline: {e}")
                # Fallback to 4 clusters if model is missing
                obs_dim += 4
            
        # Override observation space to be the embedding space (+ clustering context)
        self.observation_space = gym.spaces.Box(
            low=-1e5, high=1e5, shape=(obs_dim,), dtype=np.float32
        )
        self.success_threshold = 0.5
        
        # Initialize step counter from completed_steps to maintain global progress on resume
        self.total_steps = self.config.get('ppo', {}).get('completed_steps', 0) 
        
        # Load Cluster Biases (Translation Matrix)
        self.bias_matrix = None
        if self.clustering_config and "cluster_biases" in self.clustering_config:
            self.bias_matrix = np.array(self.clustering_config["cluster_biases"], dtype=np.float32)
            logger.info(f"Cluster-Conditioned Translation Matrix loaded. Shape: {self.bias_matrix.shape}")
            
        self.current_cluster_probs = None
        
        # Cluster Exclusion Configuration
        self.exclude_clusters = []
        if self.clustering_config and "exclude_clusters" in self.clustering_config:
            self.exclude_clusters = list(self.clustering_config["exclude_clusters"])
            logger.info(f"Loaded cluster exclusion list: {self.exclude_clusters}")

        # Dynamic Noise Scaling Configuration
        import collections
        self.dynamic_noise_enabled = False
        self.exploration_floor = 0.3
        self.exploration_ceiling = 1.8
        self.asr_update_window = 4096
        
        if self.clustering_config and "dynamic_noise" in self.clustering_config:
            dn_cfg = self.clustering_config["dynamic_noise"]
            self.dynamic_noise_enabled = dn_cfg.get("enabled", False)
            self.exploration_floor = float(dn_cfg.get("exploration_floor", 0.3))
            self.exploration_ceiling = float(dn_cfg.get("exploration_ceiling", 1.8))
            self.asr_update_window = int(dn_cfg.get("asr_update_window", 4096))
            logger.info(f"Dynamic Noise Scaling config: enabled={self.dynamic_noise_enabled}, "
                        f"floor={self.exploration_floor}, ceiling={self.exploration_ceiling}, "
                        f"window={self.asr_update_window}")
            
        self.asr_tracker = collections.deque(maxlen=self.asr_update_window)
        self.last_raw_actions = None
        self.last_conditioned_actions = None

    def _get_conditioned_observations(self, embeddings):
        """
        Batched context injection for vectorized observations.
        """
        if self.clustering_config is None:
            return embeddings.cpu().numpy().astype(np.float32)
            
        # Pipeline is now eagerly loaded in __init__
        embeddings_np = embeddings.cpu().numpy()
        cluster_probs = self.clustering_pipeline.predict_proba(embeddings_np).astype(np.float32)
        
        # Concatenate and cast to float32
        unified_obs = np.concatenate([embeddings_np, cluster_probs], axis=1)
        return unified_obs.astype(np.float32)

    def _filter_excluded_clusters(self, obs, cond_obs, indices_to_check, infos=None):
        """
        Checks if the dominant cluster of the starting observation for the specified environment indices
        is in the exclude_clusters list. If so, recursively resets them until a valid sample is secured.
        Updates obs, cond_obs, and the underlying venv's buf_obs.
        """
        if not self.exclude_clusters or self.clustering_config is None:
            return obs, cond_obs

        for i in indices_to_check:
            retries = 0
            max_retries = 100
            
            # Dominant cluster: argmax of soft probabilities (which are after embedding size 160)
            k_dom = int(np.argmax(cond_obs[i, 160:]))
            
            while k_dom in self.exclude_clusters and retries < max_retries:
                retries += 1
                logger.info(f"Worker {i}: Sample belonged to excluded cluster {k_dom}. Discarding and resetting (Retry {retries}/{max_retries}).")
                
                # Reset environment i
                res = self.venv.env_method("reset", indices=i)
                # env_method("reset") returns [(raw_audio, info)]
                raw_audio_i, info_i = res[0]
                
                # Get AASIST3 score and embedding for the new sample
                waveform_i = torch.from_numpy(raw_audio_i).unsqueeze(0).unsqueeze(1).float()
                _, embedding_i = self.detector.get_score_and_embedding(waveform_i)
                cond_obs_i = self._get_conditioned_observations(embedding_i)[0]
                
                # Update variables
                obs[i] = raw_audio_i
                cond_obs[i] = cond_obs_i
                k_dom = int(np.argmax(cond_obs_i[160:]))
                
                if infos is not None and i < len(infos):
                    infos[i].update(info_i)
                
                # Sync raw audio in venv.buf_obs
                if hasattr(self.venv, "buf_obs"):
                    for key in self.venv.buf_obs.keys():
                        if key is None:
                            self.venv.buf_obs[None][i] = raw_audio_i
                        else:
                            self.venv.buf_obs[key][i] = raw_audio_i
                            
            if retries >= max_retries:
                logger.warning(f"Worker {i}: Reached max retries ({max_retries}) trying to filter excluded clusters. Moving on to prevent stall.")

        return obs, cond_obs

    def reset(self):
        """Batched reset for all environments."""
        obs = self.venv.reset()
        # obs is [N_ENVS, 64600]
        waveform = torch.from_numpy(obs).unsqueeze(1).float() # [N, 1, L]
        
        _, embeddings = self.detector.get_score_and_embedding(waveform)
        cond_obs = self._get_conditioned_observations(embeddings)
        
        if self.exclude_clusters:
            obs, cond_obs = self._filter_excluded_clusters(obs, cond_obs, range(self.num_envs))
        
        if self.clustering_config is not None:
            # Store probabilities for step_async biasing [N, K]
            self.current_cluster_probs = cond_obs[:, 160:].astype(np.float32)
            
        return cond_obs.astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        """Override step_async to apply dynamic noise scaling and cluster-conditioned biasing before passing to base envs."""
        raw_actions = actions.copy().astype(np.float32)
        
        # 1. Compute dynamic noise scaling
        if self.dynamic_noise_enabled:
            import collections
            asr_cache = {}
            if len(self.asr_tracker) > 0:
                total_counts = collections.defaultdict(int)
                success_counts = collections.defaultdict(int)
                for k, success in self.asr_tracker:
                    total_counts[k] += 1
                    if success:
                        success_counts[k] += 1
                for k in total_counts:
                    asr_cache[k] = success_counts[k] / total_counts[k]
            
            omegas = []
            for i in range(self.num_envs):
                # get dominant cluster for env i
                if self.current_cluster_probs is not None:
                    k = int(np.argmax(self.current_cluster_probs[i]))
                else:
                    k = 0
                
                asr_k = asr_cache.get(k, 0.0)
                omega_k = np.clip(
                    self.exploration_ceiling * (1.0 - asr_k) + self.exploration_floor,
                    self.exploration_floor,
                    self.exploration_ceiling
                )
                omegas.append(omega_k)
            
            omegas = np.array(omegas, dtype=np.float32)[:, np.newaxis]
            conditioned_actions = np.clip(raw_actions * omegas, -1.0, 1.0).astype(np.float32)
        else:
            conditioned_actions = raw_actions.copy()
            
        # 2. Apply bias matrix if enabled
        if self.bias_matrix is not None and self.current_cluster_probs is not None:
            dynamic_biases = np.matmul(self.current_cluster_probs, self.bias_matrix).astype(np.float32)
            final_actions = np.clip(conditioned_actions + dynamic_biases, -1.0, 1.0).astype(np.float32)
        else:
            final_actions = conditioned_actions.astype(np.float32)
            
        self.last_raw_actions = raw_actions
        self.last_conditioned_actions = conditioned_actions
        
        self.venv.step_async(final_actions)

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
        
        # Process conditioned observations (batch)
        all_conditioned_obs = self._get_conditioned_observations(embeddings)
        
        # Apply cluster filtering for newly reset environments
        if self.exclude_clusters:
            reset_indices = [i for i, done in enumerate(dones) if done]
            if reset_indices:
                obs, all_conditioned_obs = self._filter_excluded_clusters(obs, all_conditioned_obs, reset_indices, infos)

        # Extract cluster probabilities [Total_InFERENCE_Size, K]
        all_cluster_probs = None
        if self.clustering_config is not None:
            all_cluster_probs = all_conditioned_obs[:, 160:].astype(np.float32)
            
            # Periodically log cluster distribution for diagnostic transparency
            if self.total_steps % 100 == 0:
                cluster_ids = np.argmax(all_cluster_probs, axis=1)
                unique, counts = np.unique(cluster_ids, return_counts=True)
                dist = dict(zip(unique, counts))
                logger.info(f"Batch Cluster Distribution: {dist}")

        new_rewards = []
        new_dones = []
        
        for i in range(self.num_envs):
            # If the environment finished in the worker, the reward for the STEP 
            # that just finished must come from the 'terminal_observation'.
            if i in terminal_map:
                idx = terminal_map[i]
                score = scores[idx]
                # Replace terminal raw audio with conditioned embedding to prevent crash in SB3
                infos[i]["terminal_observation"] = all_conditioned_obs[idx].astype(np.float32)
                c_probs = all_cluster_probs[idx] if all_cluster_probs is not None else None
            else:
                score = scores[i]
                c_probs = all_cluster_probs[i] if all_cluster_probs is not None else None
            
            # ASR TRACKING UPDATE
            if self.current_cluster_probs is not None:
                prev_cluster_id = int(np.argmax(self.current_cluster_probs[i]))
                is_success = bool(score > self.success_threshold)
                self.asr_tracker.append((prev_cluster_id, is_success))
            
            # Extract worker-specific info for logging
            self.current_worker = infos[i].get('worker_id', 'N/A')
            self.current_seed = infos[i].get('seed', 'N/A')

            # Update current_step for logging in reward_logic
            self.current_step = self.total_steps
            # Extract DSP parameters used in this worker's step
            dsp_params = infos[i].get('dsp_params', None)
            
            # Compute reward and check for termination based on the CORRECT score and cluster
            reward, terminated, bonus = compute_attack_reward(score, self, dsp_params=dsp_params, cluster_probs=c_probs)
            
            new_rewards.append(float(reward))
            # SB3 VecEnv handles 'dones' (terminated or truncated)
            # We add our own 'terminated' condition from the detector
            is_done = dones[i] or terminated
            new_dones.append(is_done)
            
            # If the episode ended (either in worker or here), ensure terminal_observation exists
            if is_done and "terminal_observation" not in infos[i]:
                infos[i]["terminal_observation"] = all_conditioned_obs[i].astype(np.float32)

            # Update info dictionaries with actual detector results
            infos[i]['score'] = float(score)
            infos[i]['reward'] = float(reward)
            infos[i]['bonus'] = bonus
            infos[i]['terminated'] = terminated
            if c_probs is not None:
                infos[i]['cluster_id'] = int(np.argmax(c_probs))
                
            # Save raw and conditioned actions in info dict
            if self.last_raw_actions is not None:
                infos[i]['raw_action'] = self.last_raw_actions[i]
            if self.last_conditioned_actions is not None:
                infos[i]['conditioned_action'] = self.last_conditioned_actions[i]
                
        # Update current clusters probabilities for the next step_async call [N_ENVS, K]
        if all_cluster_probs is not None:
            self.current_cluster_probs = all_cluster_probs[:self.num_envs].astype(np.float32)
            
        # Return conditioned embeddings of current observations [0:num_envs]
        return all_conditioned_obs[:self.num_envs].astype(np.float32), np.array(new_rewards, dtype=np.float32), np.array(new_dones), infos
