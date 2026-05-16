"""
Main Training Orchestrator for the KON-Artist RL Agent.

This script manages the end-to-end Reinforcement Learning pipeline:
1. Loads the AASIST3 detector as a reward/state provider.
2. Initializes the Gymnasium attack environment.
3. Configures the PPO (Proximal Policy Optimization) agent.
4. Executes training with custom callbacks for local CSV logging and W&B tracking.
5. Saves the resulting agent for evaluation.
"""
import os
import logging
import warnings
import sys
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.aasist import AASISTWrapper
from src.env.audio_attack import AudioAttackEnv
from src.env.wrappers import VecDetectorWrapper
from src.utils.callbacks import RewardLoggerCallback, WandbAudioCallback
from src.utils.logger import get_logger
from src.utils.misc import create_timestamp

# Suppress the specific FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

def load_config(config_path="configs/train_config.yaml"):
    """Loads training configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Initialize configuration
cfg = load_config()

# Initialize logger for the current module
logger = get_logger(name=__file__,
                    log_file=os.path.join(cfg['logging']['log_dir'], "train_session.log"),
                    level=logging.INFO)

# TRAIN SETTINGS
# # Colab 5h limit: 70,560 to 88,200 steps max (17~21 iterations for 4096 n_steps)
# TOTAL_UPDATES = 20  # Total number of PPO updates (each update processes N_STEPS)
# N_STEPS = 4096  # 2048 is PPO's default rollout buffer size; adjust if using a custom buffer implementation
# EPOCHS = 15  # Number of epochs per PPO update (default is 4 in stable-baselines3)
# BATCH_SIZE = 256 # Batch size for PPO updates (default is 64 in stable-baselines3, but can be adjusted based on memory constraints)
# # Automatically scale workers based on CPU cores (All cores - 1 to leave room for the main process)
# N_ENVS = max(1, os.cpu_count() - 1)  # Number of parallel environments (CPU workers)

# # PPO SCALING RULE: 
# # Increasing N_ENVS increases sample diversity per gradient update, which usually allows for more stable learning
# # but requires recalculating total timesteps to keep benchmarking comparable.
# TOTAL_ROLLOUT_BUFFER = N_STEPS * N_ENVS  # Total samples collected per PPO update
# TOTAL_TIMESTEPS = TOTAL_ROLLOUT_BUFFER * TOTAL_UPDATES  # Total timesteps for training
# TOTAL_PASSES = TOTAL_UPDATES * EPOCHS  # Total passes through the data (for logging purposes)
# BATCH_SIZE = min(BATCH_SIZE, N_STEPS * N_ENVS) # Ensure batch size does not exceed the number of steps in the buffer
# # Log the actual batch size being used after adjustment

# DERIVED SETTINGS & AUTO-SCALING
N_ENVS = cfg['env']['n_envs']
if N_ENVS == -1:
    N_ENVS = max(1, os.cpu_count() - 1)

TOTAL_ROLLOUT_BUFFER = cfg['ppo']['n_steps'] * N_ENVS
TOTAL_TIMESTEPS = TOTAL_ROLLOUT_BUFFER * cfg['ppo']['total_updates']
BATCH_SIZE = min(cfg['ppo']['batch_size'], TOTAL_ROLLOUT_BUFFER)

logger.info(f"PPO Configuration: RolloutBuffer={TOTAL_ROLLOUT_BUFFER}, MiniBatch={BATCH_SIZE}, TotalSteps={TOTAL_TIMESTEPS}")

def make_env(rank: int, seed: int = 42):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        # Lazy initialization via audio_config for pickling compatibility
        audio_config = {
            "split": cfg['audio']['split'],
            "seed": seed + rank,
            "rank": rank,
            "buffer_size": cfg['audio']['buffer_size']
        }
        # Worker has NO detector (it's in the VecDetectorWrapper on the main process)
        env = AudioAttackEnv(detector=None, audio_config=audio_config)
        # Set environment specific thresholds/limits from config
        env.success_threshold = cfg['env']['success_threshold']
        env.step_limit = cfg['env']['step_limit']
        
        # Monitor is required for RewardLoggerCallback to access ep_info_buffer
        return Monitor(env)
    return _init

def train():
    """
    Main execution loop for training the KON-Artist agent.
    """
    try:
        logger.info("===== Starting KON-Artist Training Session =====")
        
        # 1. Initialize the detector wrapper (Centralized for batched GPU inference)
        logger.info(f"Loading {cfg['model']['detector_name']} model for batched inference.")
        detector = AASISTWrapper(cfg['model']['detector_name'], device=cfg['model']['device'])
        
        # 2. Initialize Vectorized Environments
        logger.info(f"Instantiating {N_ENVS} parallel environments.")
        if N_ENVS > 1:
            env = SubprocVecEnv([make_env(i, cfg['env']['seed']) for i in range(N_ENVS)])
        else:
            env = DummyVecEnv([make_env(0, cfg['env']['seed'])])

        # 3. Wrap for Batched GPU Inference
        logger.info("Wrapping environment with VecDetectorWrapper for GPU batching.")
        env = VecDetectorWrapper(env, detector)
        env.total_timesteps = TOTAL_TIMESTEPS
        env.success_threshold = cfg['env']['success_threshold']

        # 4. Agent Instantiation
        logger.info("Configuring PPO agent.")
        model = PPO(
            policy=cfg['ppo']['policy'],
            env=env,
            n_steps=cfg['ppo']['n_steps'],
            batch_size=BATCH_SIZE,
            n_epochs=cfg['ppo']['epochs'],
            learning_rate=cfg['ppo']['learning_rate'],
            verbose=cfg['ppo']['verbose'],
            tensorboard_log=cfg['logging']['tensorboard_log']
        )
        
        # 5. Callbacks
        timestamp = create_timestamp()
        metadata = {
            "n_envs": N_ENVS,
            "architecture": cfg['model']['detector_name'],
            "timestamp": timestamp,
            "config": cfg,
            "total_timesteps": TOTAL_TIMESTEPS
        }
        reward_callback = RewardLoggerCallback(
            check_freq=cfg['logging']['reward_log_freq'], 
            id=timestamp, 
            log_dir=cfg['logging']['log_dir']
        )
        wandb_callback = WandbAudioCallback(config=metadata)
        
        # 6. Training Execution
        logger.info(f"Beginning training: TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, BATCH_TOTAL={TOTAL_ROLLOUT_BUFFER}")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[reward_callback, wandb_callback]
        )
        
        logger.info("Training finished. Saving model.")
        model_path = os.path.join(cfg['logging']['log_dir'], "kon_artist_agent")
        model.save(model_path)

    except Exception:
        logger.exception("Training failed with a critical error:")
    finally:
        if 'env' in locals():
            env.close()
        
if __name__ == "__main__":
    train()

# USAGE:
# python -m scripts.train
