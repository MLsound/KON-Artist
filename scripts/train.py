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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.models.aasist import AASISTWrapper
from src.env.audio_attack import AudioAttackEnv
from src.env.wrappers import VecDetectorWrapper
from src.utils.callbacks import RewardLoggerCallback, WandbAudioCallback, EntropyDecayCallback
from src.utils.logger import get_logger
from src.utils.misc import create_timestamp

# Suppress the specific FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Initialize logger for the current module
logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.INFO)  # Set to DEBUG for detailed trace during training

# TRAIN SETTINGS
# Colab 5h limit: 70,560 to 88,200 steps max (17~21 iterations for 4096 n_steps)
TOTAL_UPDATES = 20  # Total number of PPO updates (each update processes N_STEPS)
N_STEPS = 4096  # 2048 is PPO's default rollout buffer size; adjust if using a custom buffer implementation
EPOCHS = 15  # Number of epochs per PPO update (default is 4 in stable-baselines3)
BATCH_SIZE = 256 # Batch size for PPO updates (default is 64 in stable-baselines3, but can be adjusted based on memory constraints)a
# Automatically scale workers based on CPU cores (All cores - 1 to leave room for the main process)
N_ENVS = max(1, os.cpu_count() - 1)  # Number of parallel environments (CPU workers)

# PPO SCALING RULE: 
# Increasing N_ENVS increases sample diversity per gradient update, which usually allows for more stable learning
# but requires recalculating total timesteps to keep benchmarking comparable.
TOTAL_ROLLOUT_BUFFER = N_STEPS * N_ENVS  # Total samples collected per PPO update
TOTAL_TIMESTEPS = TOTAL_ROLLOUT_BUFFER * TOTAL_UPDATES  # Total timesteps for training
TOTAL_PASSES = TOTAL_UPDATES * EPOCHS  # Total passes through the data (for logging purposes)
BATCH_SIZE = min(BATCH_SIZE, TOTAL_ROLLOUT_BUFFER) # Ensure batch size does not exceed the number of steps in the buffer

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

# Log the actual batch size being used after adjustment
logger.info(f"PPO Configuration: RolloutBuffer={TOTAL_ROLLOUT_BUFFER}, MiniBatch={BATCH_SIZE}, TotalSteps={TOTAL_TIMESTEPS}")

def make_env(rank: int, seed: int = 42):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        # Lazy initialization via audio_config for pickling compatibility
        audio_config = {
            "split": "train",
            "seed": seed + rank,
            "buffer_size": 1000
        }
        # Worker has NO detector (it's in the VecDetectorWrapper on the main process)
        env = AudioAttackEnv(detector=None, audio_config=audio_config)
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
        logger.info("Loading AASIST3 model for batched inference.")
        detector = AASISTWrapper("MTUCI/AASIST3", device="cuda")
        
        # 2. Initialize Vectorized Environments
        logger.info(f"Instantiating {N_ENVS} parallel environments.")
        if N_ENVS > 1:
            # SubprocVecEnv runs environments in separate CPU processes
            env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
        else:
            # Fallback to DummyVecEnv for single environment debugging
            env = DummyVecEnv([make_env(0)])

        # 3. Wrap for Batched GPU Inference
        # This wrapper captures raw audio from workers and runs AASIST3 in batches
        logger.info("Wrapping environment with VecDetectorWrapper for GPU batching.")
        env = VecDetectorWrapper(env, detector)

        # 4. Agent Instantiation
        logger.info("Configuring PPO agent.")
        # Learning Rate Schedule (Linear Decay)
        lr_schedule = linear_schedule(3e-4)
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr_schedule,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=EPOCHS,
            verbose=1,
            tensorboard_log="outputs/tensorboard/"
        )
        
        # 5. Callbacks
        timestamp = create_timestamp()
        metadata = {
            "n_envs": N_ENVS,
            "architecture": "AASIST3",
            "timestamp": timestamp,
            "epochs": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "n_steps": N_STEPS,
            "total_timesteps": TOTAL_TIMESTEPS
        }
        reward_callback = RewardLoggerCallback(check_freq=1, id=timestamp, log_dir="outputs/")
        wandb_callback = WandbAudioCallback(config=metadata)
        # Entropy Decay: From 0.01 (exploration) to 0.001 (exploitation)
        entropy_callback = EntropyDecayCallback(initial_ent_coef=0.01, final_ent_coef=0.001, total_timesteps=TOTAL_TIMESTEPS)
        
        # 6. Training Execution
        logger.info(f"Beginning training: TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, BATCH_TOTAL={N_STEPS * N_ENVS}")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[reward_callback, wandb_callback, entropy_callback]
        )
        
        logger.info("Training finished. Saving model.")
        model.save("outputs/kon_artist_agent")

    # except Exception as e:
    #     logger.error(f"Training failed: {e}")
    except Exception:
        # Automatically captures and logs the full traceback
        logger.exception("Training failed with a critical error:")
    finally:
         # Strictly for cleanup operations
        if 'env' in locals():
            env.close()
        
if __name__ == "__main__":
    train()

# USAGE:
# python -m scripts.train