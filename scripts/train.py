"""
Main Training Orchestrator for the KON-Artist RL Agent.

This script manages the end-to-end Reinforcement Learning pipeline:
1. Loads the AASIST3 detector as a reward/state provider.
2. Initializes the Gymnasium attack environment.
3. Configures the PPO (Proximal Policy Optimization) agent.
4. Executes training with custom callbacks for local CSV logging and W&B tracking.
5. Saves the resulting agent for evaluation.
"""

from stable_baselines3 import PPO
from src.data.loader import get_asvspoof_loader, generator_from_ds
from src.models.aasist import AASISTWrapper
from src.env.audio_attack import AudioAttackEnv
from src.utils.callbacks import RewardLoggerCallback, WandbAudioCallback
from src.utils.logger import get_logger
from src.utils.misc import create_timestamp
import logging
import warnings

# Suppress the specific FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Initialize logger for the current module
logger = get_logger(name=__file__,
                    log_file="outputs/train_session.log",
                    level=logging.INFO)  # Set to DEBUG for detailed trace during training

# TRAIN SETTINGS
# Colab 5h limit: 70,560 to 88,200 steps max (17~21 iterations for 4096 n_steps)
TOTAL_UPDATES = 5  # Total number of PPO updates (each update processes N_STEPS)
N_STEPS = 2048  # PPO's default rollout buffer size; adjust if using a custom buffer implementation
EPOCHS = 10  # Number of epochs per PPO update (default is 4 in stable-baselines3)
BATCH = 512 # Batch size for PPO updates (default is 64 in stable-baselines3, but can be adjusted based on memory constraints)

TOTAL_TIMESTEPS = N_STEPS * TOTAL_UPDATES  # Total timesteps is the product of steps per update and total updates
TOTAL_PASSES = TOTAL_UPDATES * EPOCHS  # Total passes through the data (for logging purposes)
BATCH = min(BATCH, N_STEPS)  # Ensure batch size does not exceed the number of steps in the buffer
# logger.debug(f"batch used: {BATCH}") # Log the actual batch size being used after adjustment

def train():
    """
    Main execution loop for training the KON-Artist agent.
    """
    try:
        logger.info("===== Starting KON-Artist Training Session =====")
        
        # 1. Initialize the detector wrapper and environment
        logger.info("Loading AASIST3 model and initializing environment.")
        detector = AASISTWrapper("MTUCI/AASIST3")
        
        # 2. Initialize streaming dataset for the environment
        logger.info("Initializing audio stream from ASVspoof 2019 dataset.")
        ds = get_asvspoof_loader(split="train")
        audio_stream = generator_from_ds(ds)

        # 3. Create the Gymnasium environment with the streaming dataset
        logger.info("Creating AudioAttackEnv with the loaded model.")
        env = AudioAttackEnv(
            detector=detector, 
            dsp_config={}, 
            audio_files=audio_stream # Pass the actual generator
        )

        # 4. Agent and Callback Instantiation
        logger.info("Configuring PPO agent.")
        # MlpPolicy is used to process the 160-dim embeddings
        # model = PPO("MlpPolicy", env, verbose=2)
        model = PPO(policy="MlpPolicy",
                    env=env,
                    n_steps=N_STEPS,
                    batch_size=BATCH,
                    n_epochs=EPOCHS,
                    verbose=2) # Set to 2 for maximum verbosity during training
        # verbosity:
        #   0: No output (silent).
        #   1: Basic information messages (such as the device used or wrappers applied) and training statistics.
        #   2: Maximum verbosity, which includes detailed debug messages to monitor the internal behavior of the algorithm.
        
        # 5. Callbacks for logging rewards and audio samples
        logger.info("Setting up callbacks.")
        timestamp = create_timestamp()
        metadata={
                    "architecture": "AASIST3",
                    "dataset": "ASVspoof 2019",
                    "timestamp": timestamp,
                    "epochs": EPOCHS,
                    "batch": BATCH,
                    "n_steps": N_STEPS,
                    "total_timesteps": TOTAL_TIMESTEPS,
                    "total_passes": TOTAL_PASSES
                }
        reward_callback = RewardLoggerCallback(check_freq=1, id=timestamp, log_dir="outputs/") # Logs rewards at every step to a timestamped CSV file in outputs/history/
        wandb_callback = WandbAudioCallback(config=metadata) # Weights & Biases callback to track DSP parameters and audio samples during training
        
        # 6. Training Execution
        logger.info("Beginning training.")
        logger.info(f"Training configuration: N_STEPS={N_STEPS}, EPOCHS={EPOCHS}, BATCH={BATCH}, TOTAL_UPDATES={TOTAL_UPDATES}, TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, # Set to 2048 for a single PPO update cycle (one full buffer)
            callback=[reward_callback, wandb_callback]
        )
        
        logger.info("Training finished. Data saved in outputs/rewards_history.csv")
        logger.info("Saving model.")
        model.save("outputs/kon_artist_agent")
        
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