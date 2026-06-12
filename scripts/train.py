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
import torch
import time
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

def make_env(rank: int, seed: int = 42, completed_steps: int = 0, session_total_steps: int = None):
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
        env = AudioAttackEnv(
            detector=None, 
            audio_config=audio_config, 
            bonus=cfg['ppo'].get('bonus', True), 
            bonus_amount=cfg['ppo'].get('bonus_amount', 250.0),
            completed_steps=completed_steps,
            initial_checkpoint_steps=completed_steps,
            session_total_steps=session_total_steps
        )
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
        # Unified hyperparameter log for automated documentation synchronization
        logger.info(f"TOTAL_TIMESTEPS = {TOTAL_TIMESTEPS} | N_STEPS = {cfg['ppo']['n_steps']} | TOTAL_UPDATES = {cfg['ppo']['total_updates']} | EPOCHS = {cfg['ppo']['epochs']} | BATCH = {BATCH_SIZE}")
        
        # 1. Hardware Check: Enforce CUDA if requested
        requested_device = cfg['model'].get('device', 'cpu')
        if requested_device == "cuda" and not torch.cuda.is_available():
            critical_error = "‼️ CRITICAL: CUDA requested but not available. Aborting to prevent inefficient CPU execution."
            logger.error(critical_error)
            raise RuntimeError(critical_error)
        if requested_device == "cpu":
            logger.warning("⚠️ WARNING: CPU execution requested. Training will be significantly slower. Ensure this is intentional. (Go to configs/train_config.yaml to change this setting)")
            
        # 2. Initialize the detector wrapper (Centralized for batched GPU inference)
        logger.info(f"Loading {cfg['model']['detector_name']} model for batched inference.")
        
        # Log Bonus Configuration
        bonus_enabled = cfg['ppo'].get('bonus', True)
        bonus_amount = cfg['ppo'].get('bonus_amount', 250.0)
        logger.info(f"Bonus Reward: {f'ENABLED (Amount: {bonus_amount})' if bonus_enabled else '❕DISABLED'}")

        detector = AASISTWrapper(cfg['model']['detector_name'], device=requested_device)
        
        # Initialize learning rate schedule and tensorboard log directory early
        lr_schedule = linear_schedule(cfg['ppo']['learning_rate'])
        tensorboard_log_dir = cfg['logging']['tensorboard_log']

        # CHECKPOINT DETECTION (Moved early to inform wrappers/logic)
        checkpoint_cfg = cfg['logging'].get('checkpoint', {})
        latest_checkpoint = None
        completed_steps = 0
        model = None
        if checkpoint_cfg.get('load_last', False):
            latest_checkpoint = get_latest_checkpoint(checkpoint_cfg.get('save_path', "outputs/checkpoints/"))
            if latest_checkpoint:
                try:
                    # Load PPO checkpoint early to extract genuine completed step count
                    model = PPO.load(
                        latest_checkpoint,
                        device=requested_device,
                        custom_objects={"learning_rate": lr_schedule},
                        tensorboard_log=tensorboard_log_dir
                    )
                    completed_steps = int(model.num_timesteps)
                    logger.info(f"Checkpoint detected: {completed_steps} genuine steps already completed.")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint early: {e}. Falling back to fresh agent creation.")
                    latest_checkpoint = None
                    model = None

        pending_timesteps = TOTAL_TIMESTEPS - completed_steps
        cfg['ppo']['pending_timesteps'] = pending_timesteps
        cfg['ppo']['completed_steps'] = completed_steps

        # 2. Initialize Vectorized Environments
        logger.info(f"Instantiating {N_ENVS} parallel environments.")
        if N_ENVS > 1:
            env = SubprocVecEnv([make_env(i, cfg['env']['seed'], completed_steps, pending_timesteps) for i in range(N_ENVS)])
        else:
            env = DummyVecEnv([make_env(0, cfg['env']['seed'], completed_steps, pending_timesteps)])

        # 3. Wrap for Batched GPU Inference
        logger.info("Wrapping environment with VecDetectorWrapper for GPU batching.")
        
        env = VecDetectorWrapper(
            env, 
            detector, 
            bonus=cfg['ppo'].get('bonus', True), 
            bonus_amount=cfg['ppo'].get('bonus_amount', 250.0),
            config=cfg,
            initial_checkpoint_steps=completed_steps,
            session_total_steps=pending_timesteps
        )
        env.total_timesteps = TOTAL_TIMESTEPS
        env.success_threshold = cfg['env']['success_threshold']
        
        # 4. Wrap with VecMonitor to track actual rewards
        # VecMonitor is required for RewardLoggerCallback to access ep_info_buffer
        env = VecMonitor(env)

        # 5. Agent Instantiation / Setup
        if model is not None:
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            model.set_env(env)
        else:
            logger.info("Configuring PPO agent.")
            model = PPO(
                policy=cfg['ppo']['policy'],
                env=env,
                n_steps=cfg['ppo']['n_steps'],
                batch_size=BATCH_SIZE,
                n_epochs=cfg['ppo']['epochs'],
                learning_rate=lr_schedule,
                clip_range=cfg['ppo'].get('clip_range', 0.2),
                verbose=cfg['ppo']['verbose'],
                tensorboard_log=tensorboard_log_dir
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
            log_dir=cfg['logging']['log_dir'],
            initial_checkpoint_steps=completed_steps,
            session_total_steps=pending_timesteps
        )
        wandb_callback = WandbAudioCallback(
            config=metadata,
            initial_checkpoint_steps=completed_steps,
            session_total_steps=pending_timesteps
        )
        lr_callback = LearningRateLoggerCallback(
            verbose=1,
            initial_checkpoint_steps=completed_steps,
            session_total_steps=pending_timesteps
        )
        
        # Entropy Decay: From exploration to exploitation
        entropy_cfg = cfg['ppo'].get('entropy', {'initial': 0.01, 'final': 0.001})
        entropy_callback = EntropyDecayCallback(
            initial_ent_coef=entropy_cfg['initial'], 
            final_ent_coef=entropy_cfg['final'], 
            total_timesteps=TOTAL_TIMESTEPS,
            initial_checkpoint_steps=completed_steps,
            session_total_steps=pending_timesteps
        )

        # Checkpoint Callback
        save_freq_steps = checkpoint_cfg.get('save_freq', 1) * TOTAL_ROLLOUT_BUFFER
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, save_freq_steps),
            save_path=checkpoint_cfg.get('save_path', "outputs/checkpoints/"),
            name_prefix=checkpoint_cfg.get('name_prefix', "kon_artist_ppo")
        )
        
        # 6. Training Execution
        pending_timesteps = TOTAL_TIMESTEPS - completed_steps
        if pending_timesteps <= 0:
            logger.info(f"‼️ Target steps ({TOTAL_TIMESTEPS}) reached or exceeded by checkpoint ({completed_steps}). Training complete.")
            return

        estimated_time_s = pending_timesteps * cfg['logging'].get('time_per_step', 0.0)
        hours, remainder = divmod(estimated_time_s, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Log estimated time based on pending steps to provide a realistic expectation for training duration, especially when resuming from checkpoints.
        if pending_timesteps == TOTAL_TIMESTEPS:
            logger.info(f"⌛️ Estimated total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s (based on {cfg['logging'].get('time_per_step', 0.0)}s/step)")
            logger.info(f"Beginning training: TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}, BATCH_TOTAL={TOTAL_ROLLOUT_BUFFER}")
        else:
            logger.info(f"⌛️ Estimated PENDING training time: {int(hours)}h {int(minutes)}m {int(seconds)}s (based on {cfg['logging'].get('time_per_step', 0.0)}s/step)")
            logger.info(f"Beginning training: PENDING_STEPS={pending_timesteps}, TARGET_TOTAL={TOTAL_TIMESTEPS}")
            
        # --- ACTOR NETWORK BIASING (Prior Knowledge Injection) ---
        # We warm-start exploration by shifting the initial mean toward a "Winning Signature".
        biasing_cfg = cfg['ppo'].get('biasing', {})
        if not latest_checkpoint and biasing_cfg.get('enabled', False):
            logger.info(f"Injecting 'Winning Signature' bias into Actor network: {biasing_cfg['target_signature']}")
            target_signature = torch.tensor(biasing_cfg['target_signature'], dtype=torch.float32)
            
            with torch.no_grad():
                # Check if SB3 policy uses squashing (standard Tanh on DiagGaussian output)
                # PPO MlpPolicy usually doesn't, but we handle it for architectural robustness.
                use_squashing = getattr(model.policy, "squash_output", False)
                
                if use_squashing:
                    # Apply inverse Tanh (arctanh) so post-activation mean matches targets
                    # For 1.0, we use 4.0 to avoid infinity while forcing saturation
                    biases = torch.zeros_like(target_signature)
                    for i, val in enumerate(target_signature):
                        if val >= 1.0: biases[i] = 4.0
                        elif val <= -1.0: biases[i] = -4.0
                        else: biases[i] = torch.atanh(val)
                else:
                    biases = target_signature
                
                # Overwrite action_net bias (shape: [action_dim])
                model.policy.action_net.bias.copy_(biases)
        # ---------------------------------------------------------

        start_time = time.time()
        model.learn(
            total_timesteps=pending_timesteps,
            callback=[reward_callback, wandb_callback, checkpoint_callback, entropy_callback, lr_callback],
            reset_num_timesteps=False if latest_checkpoint else True
        )
        end_time = time.time()

        total_duration = end_time - start_time
        actual_time_per_step = total_duration / pending_timesteps
        logger.info(f"Training finished. Actual average processing time: {actual_time_per_step:.5f}s/step")
        
        logger.info("Saving model.")
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
