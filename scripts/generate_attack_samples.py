"""
Adversarial Sample Generation Script.

This script uses a trained PPO agent to extract optimized DSP configurations
for spoofed audio samples and generates modified audio files as attack samples.

Usage Example:
    python -m scripts.generate_attack_samples --model outputs/weights/agent --samples 10
"""
import os
import argparse
import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from src.models.aasist import AASISTWrapper
from src.env.audio_attack import AudioAttackEnv
from src.data.loader import get_asvspoof_loader, generator_from_ds
from src.utils.logger import get_logger
import logging

# Initialize logger
logger = get_logger(name=__file__,
                    log_file="outputs/logs/eval_session.log",
                    level=logging.INFO)

def run_generation(model_path, num_samples=10, output_dir="outputs/attack_samples/"):
    """
    Loads a trained agent and generates modified audio samples.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load AASIST3 Detector (Frozen)
    logger.info("Loading AASIST3 detector...")
    detector = AASISTWrapper("MTUCI/AASIST3", device=device)

    # 2. Load Trained RL Agent
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        logger.error(f"Trained model not found at {model_path}")
        return

    logger.info(f"Loading RL agent from {model_path}...")
    agent = PPO.load(model_path, device=device)

    # 3. Load Data Loader (Spoof samples only)
    logger.info("Loading ASVspoof 2019 validation split for generation...")
    ds = get_asvspoof_loader(split="validation")
    audio_gen = generator_from_ds(ds)

    # 4. Initialize Environment
    # Note: Using detector=detector allows tracking the score during generation
    env = AudioAttackEnv(detector=detector, audio_files=audio_gen)

    # 5. Prepare Output Directory
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    successful_attack_configs = [] # List to store DSP parameters of successful attacks

    logger.info(f"Generating {num_samples} attack samples...")
    for i in tqdm(range(num_samples)):
        obs, info = env.reset()
        initial_score = info['score']
        
        terminated = False
        truncated = False
        final_info = info
        
        # Run the agent rollout
        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info
            
        final_score = final_info['score']
        dsp_params = final_info['dsp_params']
        is_success = bool(final_score > 0.5)
        
        # Re-generate the modified audio to save it
        modified_audio = env.dsp(env.current_audio, dsp_params)
        
        # 6. Save Modified Audio
        filename = f"attack_sample_{i:03d}_score_{final_score:.4f}.wav"
        save_path = os.path.join(output_dir, filename)
        
        # Save as 16kHz mono (AASIST3 standard)
        torchaudio.save(save_path, modified_audio.cpu(), 16000)
        
        # Track metadata
        metadata.append({
            "sample_index": i,
            "filename": filename,
            "initial_score": float(initial_score),
            "final_score": float(final_score),
            "success": is_success,
            "dsp_configurations": dsp_params
        })

        # Store parameters for successful attacks for further analysis
        if is_success:
            successful_attack_configs.append({
                "sample_index": i,
                "score": float(final_score),
                "dsp_params": dsp_params
            })

    # 7. Save Metadata and Successful Configurations
    metadata_path = os.path.join(output_dir, "generation_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    success_configs_path = os.path.join(output_dir, "successful_attack_configs.json")
    with open(success_configs_path, "w") as f:
        json.dump(successful_attack_configs, f, indent=4)

    logger.info(f"Generation complete. Metadata and {len(successful_attack_configs)} success configs saved to {output_dir}")
    print(f"\nSuccessfully generated {num_samples} samples.")
    print(f"Recorded {len(successful_attack_configs)} successful attacks in {success_configs_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attack samples using a trained RL agent.")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the trained PPO agent zip file.")
    parser.add_argument("--samples", type=int, default=10, 
                        help="Number of samples to generate.")
    parser.add_argument("--out", type=str, default="outputs/attack_samples/", 
                        help="Directory to save generated samples.")
    
    args = parser.parse_args()
    run_generation(args.model, args.samples, args.out)

# USAGE:
# python -m scripts.generate_attack_samples --model outputs/weights/kon_artist_agent_20260425_142549 --samples 20
