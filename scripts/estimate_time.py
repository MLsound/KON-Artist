#!/usr/bin/env python3
"""
Utility script to estimate the total training time for KON-Artist 
based on the current configuration in `configs/train_config.yaml`.

# Usage:
    python -m scripts.estimate_time --config configs/train_config.yaml --colab --start-step 50000

"""
import yaml
import os

def load_config(config_path="configs/train_config.yaml"):
    """Loads training configuration from a YAML file."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        exit(1)
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

def estimate_time(config_path="configs/train_config.yaml", manual_workers=None, use_checkpoint=False, start_step=None):
    cfg = load_config(config_path)

    # Automatically scale workers based on CPU cores, or use manual override
    if manual_workers is not None:
        n_envs = int(manual_workers)
    else:
        n_envs = int(cfg['env']['n_envs'])
        if n_envs == -1:
            n_envs = max(1, os.cpu_count() - 1)

    # Calculate Timesteps
    n_steps = int(cfg['ppo']['n_steps'])
    total_updates = int(cfg['ppo']['total_updates'])
    
    total_rollout_buffer = n_steps * n_envs
    total_timesteps = total_rollout_buffer * total_updates

    completed_steps = 0
    checkpoint_msg = ""
    
    if start_step is not None:
        completed_steps = int(start_step)
        checkpoint_msg = f" (Manually starting from {completed_steps} steps)"
    elif use_checkpoint:
        save_path_str = str(cfg['logging'].get('checkpoint', {}).get('save_path', "outputs/checkpoints/"))
        latest_checkpoint = get_latest_checkpoint(save_path_str)
        if latest_checkpoint:
            try:
                filename = os.path.basename(latest_checkpoint)
                completed_steps = int(filename.split('_')[-2])
                checkpoint_msg = f" (Resuming from {completed_steps} steps auto-detected)"
            except (ValueError, IndexError):
                print("Warning: Could not extract step count from checkpoint.")
        else:
            print("Warning: No checkpoint found. Estimating full time.")

    pending_timesteps = max(0, total_timesteps - completed_steps)

    # Get estimated time per step
    time_per_step = float(cfg['logging'].get('time_per_step', 0.18415))

    # Calculate Total Time
    estimated_time_s = pending_timesteps * time_per_step
    hours, remainder = divmod(estimated_time_s, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("=" * 50)
    print("KON-Artist Training Time Estimator")
    print("=" * 50)
    print(f"Configuration File : {config_path}")
    print(f"Workers (n_envs)   : {n_envs}")
    print(f"Rollout Buffer     : {total_rollout_buffer} ({n_steps} steps * {n_envs} envs)")
    print(f"Total Updates      : {total_updates}")
    print(f"Total Timesteps    : {total_timesteps}{checkpoint_msg}")
    if completed_steps > 0:
        print(f"Pending Timesteps  : {pending_timesteps}")
    print(f"Time Per Step      : {time_per_step:.5f}s")
    print("-" * 50)
    print(f"Estimated Time     : {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    print("=" * 50)
    if completed_steps == 0:
        print("Note: This assumes starting from step 0. Use --checkpoint to calculate remaining time.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimate KON-Artist training time.")
    parser.add_argument('-c', '--config', default='configs/train_config.yaml', help='Path to config file')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Manually override the number of workers')
    parser.add_argument('--colab', action='store_true', help='Shortcut to set workers to 1 for Colab GPU')
    parser.add_argument('--checkpoint', action='store_true', help='Calculate remaining time based on the latest checkpoint')
    parser.add_argument('--start-step', type=int, default=None, help='Manually set the number of steps already completed')
    args = parser.parse_args()
    
    workers = 1 if args.colab else args.workers
    
    estimate_time(args.config, workers, args.checkpoint, args.start_step)
