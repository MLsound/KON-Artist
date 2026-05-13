"""
Generalization Testing and Performance Evaluation Script.

This script evaluates a trained RL agent against the AASIST3 detector 
using the ASVspoof 2019 test split. It calculates:
1. Baseline EER: Detector performance on original samples.
2. Adversarial EER: Detector performance on samples modified by the RL agent.
3. Attack Success Rate (ASR): Percentage of spoofed samples that fooled the detector.
"""

import os
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from src.models.aasist import AASISTWrapper
from src.env.audio_attack import AudioAttackEnv
from src.data.loader import get_asvspoof_loader, eval_generator
from src.utils.metrics import compute_eer
from src.utils.logger import get_logger
import logging

# Initialize logger
logger = get_logger(name=__file__,
                    log_file="outputs/eval_session.log",
                    level=logging.INFO)

def run_evaluation(model_path, num_samples=500):
    """
    Executes the evaluation pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting evaluation on {device}...")

    # 1. Load AASIST3 Detector
    logger.info("Loading AASIST3 detector...")
    detector = AASISTWrapper("MTUCI/AASIST3", device=device)

    # 2. Load Trained Agent
    if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
        logger.info(f"Loading RL agent from {model_path}...")
        agent = PPO.load(model_path)
    else:
        logger.warning(f"Agent not found at {model_path}. Running baseline only.")
        agent = None

    # 3. Load Test Dataset
    logger.info("Loading ASVspoof 2019 test split...")
    # Using 'test' split for final generalization verification
    ds = get_asvspoof_loader(split="test")
    test_gen = eval_generator(ds)

    # 4. Collection Buffers
    bonafide_scores = []
    baseline_spoof_scores = []
    adversarial_spoof_scores = []
    
    # Initialize environment for adversarial evaluation
    # We pass None for audio_files initially as we will manually provide samples
    env = AudioAttackEnv(detector=detector, dsp_config={}, audio_files=[])

    logger.info(f"Processing {num_samples} samples...")
    
    count = 0
    success_count = 0
    spoof_count = 0
    
    for audio, label in test_gen:
        if count >= num_samples:
            break
            
        # label 1: Bonafide, 0: Spoof
        if label == 1:
            # Get score for bonafide sample
            scores, _ = detector.get_score_and_embedding(audio)
            bonafide_scores.append(scores[0])
        else:
            spoof_count += 1
            # Get baseline score for spoof sample
            scores, _ = detector.get_score_and_embedding(audio)
            baseline_score = scores[0]
            baseline_spoof_scores.append(baseline_score)
            
            if agent is not None:
                # Run adversarial attack
                # Manually set environment state for the current audio
                env.current_audio = audio
                env.current_step = 0
                
                # Get initial observation
                _, embeddings = detector.get_score_and_embedding(audio)
                obs = env._get_obs(embeddings[0])
                
                # Agent roll-out (limited by env.step_limit)
                terminated = False
                truncated = False
                final_score = baseline_score
                
                while not (terminated or truncated):
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    final_score = info['score']
                
                adversarial_spoof_scores.append(final_score)
                if final_score > 0.5:
                    success_count += 1
            else:
                adversarial_spoof_scores.append(baseline_score)

        count += 1
        if count % 50 == 0:
            logger.info(f"Progress: {count}/{num_samples}")

    # 5. Compute Metrics
    logger.info("Computing metrics...")
    
    bonafide_scores = np.array(bonafide_scores)
    baseline_spoof_scores = np.array(baseline_spoof_scores)
    adversarial_spoof_scores = np.array(adversarial_spoof_scores)
    
    # Baseline EER
    base_eer, base_thresh = compute_eer(bonafide_scores, baseline_spoof_scores)
    
    # Adversarial EER
    adv_eer, adv_thresh = compute_eer(bonafide_scores, adversarial_spoof_scores)
    
    # Attack Success Rate (ASR)
    asr = (success_count / spoof_count) * 100 if spoof_count > 0 else 0
    
    # 6. Report Generation
    print("\n" + "="*40)
    print("KON-ARTIST EVALUATION REPORT")
    print("="*40)
    print(f"Samples Evaluated: {count} (Bonafide: {len(bonafide_scores)}, Spoof: {spoof_count})")
    print("-" * 40)
    print(f"Baseline EER:      {base_eer*100:.4f}%")
    print(f"Adversarial EER:   {adv_eer*100:.4f}%")
    print(f"Degradation:       {(adv_eer - base_eer)*100:.4f}% (absolute)")
    print("-" * 40)
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    print("="*40)
    
    # Log results
    logger.info(f"Evaluation Complete. Baseline EER: {base_eer*100:.2f}%, Adversarial EER: {adv_eer*100:.2f}%, ASR: {asr:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KON-Artist RL Agent generalization.")
    parser.add_argument("--model", type=str, default="outputs/kon_artist_agent_vectorized", 
                        help="Path to the trained PPO agent.")
    parser.add_argument("--samples", type=int, default=500, 
                        help="Number of samples to evaluate.")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.samples)
