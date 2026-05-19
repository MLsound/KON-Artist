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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.aasist import AASISTWrapper
from src.env.audio_attack import AudioAttackEnv
from src.env.wrappers import VecDetectorWrapper
from src.utils.callbacks import RewardLoggerCallback, WandbAudioCallback, EntropyDecayCallback, LearningRateLoggerCallback
from src.utils.logger import get_logger
from src.utils.misc import create_timestamp

# Suppress the specific FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

def load_config(config_path="configs/train_config.yaml"):
    """Loads training configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_latest_checkpoint(checkpoint_dir):
    """Finds the most recent checkpoint file in the given directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not checkpoints:
        return None
    # Sort by modification time to get the latest one
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: (float) Initial learning rate.
    :return: (function)
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value
    return func

# Initialize configuration
cfg = load_config() # Load training configuration from YAML file

# Initialize logger for the current module
logger = get_logger(name=__file__,
                    log_file=os.path.join(cfg['logging']['log_dir'], "train_session.log"),
                    level=logging.INFO)

# DERIVED SETTINGS & AUTO-SCALING
# Automatically scale workers based on CPU cores (All cores - 1 to leave room for the main process)
N_ENVS = cfg['env']['n_envs']
# Increasing N_ENVS increases sample diversity per gradient update, which usually allows for more stable learning
# but requires recalculating total timesteps to keep benchmarking comparable.
if N_ENVS == -1:
    N_ENVS = max(1, os.cpu_count() - 1) # Number of parallel environments (CPU workers)

# PPO SCALING RULE: Total timesteps should be divisible by (n_steps * n_envs) to ensure full batches during updates.
TOTAL_ROLLOUT_BUFFER = cfg['ppo']['n_steps'] * N_ENVS # Total samples collected per PPO update (must be divisible by batch_size)
TOTAL_TIMESTEPS = TOTAL_ROLLOUT_BUFFER * cfg['ppo']['total_updates'] # Total timesteps for training (must be divisible by batch_size)
BATCH_SIZE = min(cfg['ppo']['batch_size'], TOTAL_ROLLOUT_BUFFER) # Ensure batch size does not exceed the number of steps in the buffer

# Log the actual training configuration for transparency and debugging
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
        
        return env
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
        
        # 4. Wrap with VecMonitor to track actual rewards
        # VecMonitor is required for RewardLoggerCallback to access ep_info_buffer
        env = VecMonitor(env)

        # 5. Agent Instantiation
        logger.info("Configuring PPO agent.")
        
        checkpoint_cfg = cfg['logging'].get('checkpoint', {})
        latest_checkpoint = None
        if checkpoint_cfg.get('load_last', False):
            latest_checkpoint = get_latest_checkpoint(checkpoint_cfg.get('save_path', "outputs/checkpoints/"))
            
        # Initialize learning rate schedule
        lr_schedule = linear_schedule(cfg['ppo']['learning_rate'])

        if latest_checkpoint:
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            model = PPO.load(
                latest_checkpoint, 
                env=env,
                learning_rate=lr_schedule,
                tensorboard_log=cfg['logging']['tensorboard_log']
            )
        else:
            model = PPO(
                policy=cfg['ppo']['policy'],
                env=env,
                n_steps=cfg['ppo']['n_steps'],
                batch_size=BATCH_SIZE,
                n_epochs=cfg['ppo']['epochs'],
                learning_rate=lr_schedule,
                clip_range=cfg['ppo'].get('clip_range', 0.2),
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
            "total_timesteps": TOTAL_TIMESTEPS,
            "resumed_from": latest_checkpoint
        }
        reward_callback = RewardLoggerCallback(
            check_freq=cfg['logging']['reward_log_freq'], 
            id=timestamp, 
            log_dir=cfg['logging']['log_dir']
        )
        wandb_callback = WandbAudioCallback(config=metadata)
        lr_callback = LearningRateLoggerCallback(verbose=1)
        
        # Entropy Decay: From exploration to exploitation
        entropy_cfg = cfg['ppo'].get('entropy', {'initial': 0.01, 'final': 0.001})
        entropy_callback = EntropyDecayCallback(
            initial_ent_coef=entropy_cfg['initial'], 
            final_ent_coef=entropy_cfg['final'], 
            total_timesteps=TOTAL_TIMESTEPS
        )

        # Checkpoint Callback
        save_freq_steps = checkpoint_cfg.get('save_freq', 1) * TOTAL_ROLLOUT_BUFFER
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, save_freq_steps),
            save_path=checkpoint_cfg.get('save_path', "outputs/checkpoints/"),
            name_prefix=checkpoint_cfg.get('name_prefix', "kon_artist_ppo")
        )
        
        # 6. Training Execution
        logger.info(f"Beginning training: TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, BATCH_TOTAL={TOTAL_ROLLOUT_BUFFER}")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[reward_callback, wandb_callback, checkpoint_callback, entropy_callback, lr_callback],
            reset_num_timesteps=False if latest_checkpoint else True
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
