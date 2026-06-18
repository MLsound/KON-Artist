"""
Generalization Testing and Performance Evaluation Script.

This script evaluates a trained RL agent against the AASIST3 detector 
using the ASVspoof 2019 test split. It calculates:
1. Baseline EER: Detector performance on original samples.
2. Adversarial EER: Detector performance on samples modified by the RL agent.
3. Attack Success Rate (ASR): Percentage of spoofed samples that fooled the detector.

Usage Example:
    python -m scripts.evaluate --model outputs/weights/kon_artist_agent --samples 500
"""

import os
import argparse
import torch
import numpy as np
import time
import json
from stable_baselines3 import PPO
from src.models.aasist import AASISTWrapper
from src.env.audio_attack import AudioAttackEnv
from src.data.loader import get_asvspoof_loader, eval_generator
from src.utils.metrics import compute_eer, compute_mindcf
from src.utils.logger import get_logger
import logging

# Initialize logger
logger = get_logger(name=__file__,
                    log_file="outputs/logs/eval_session.log",
                    level=logging.INFO)

def profile_pipeline(detector, num_batches=5, batch_size=4):
    """
    Profiles the batched inference latency of the detector.
    """
    logger.info(f"Profiling pipeline with batch_size={batch_size}...")
    # Create a dummy batch of 16kHz mono audio (64600 samples ~4.03s)
    dummy_input = torch.randn(batch_size, 1, 64600).to(detector.device)
    
    # Warm-up
    for _ in range(3):
        detector.get_score_and_embedding(dummy_input)
        
    start_time = time.time()
    for _ in range(num_batches):
        detector.get_score_and_embedding(dummy_input)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_batches
    logger.info(f"Average batch latency: {avg_latency:.4f}s ({avg_latency/batch_size:.4f}s per sample)")
    return avg_latency

def run_evaluation(model_path, num_samples=500):
    """
    Executes the evaluation pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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

    # 3. Profiling
    latency = profile_pipeline(detector)

    # 4. Load Test Dataset
    logger.info("Loading ASVspoof 2019 test split...")
    ds = get_asvspoof_loader(split="test")
    test_gen = eval_generator(ds)

    # 5. Collection Buffers
    bonafide_scores = []
    baseline_spoof_scores = []
    adversarial_spoof_scores = []
    
    env = AudioAttackEnv(detector=detector, dsp_config={}, audio_files=[])

    logger.info(f"Processing up to {num_samples} balanced samples...")
    
    bonafide_target = num_samples // 2
    spoof_target = num_samples - bonafide_target
    
    bonafide_count = 0
    spoof_count = 0
    success_count = 0
    
    for audio, label in test_gen:
        if bonafide_count >= bonafide_target and spoof_count >= spoof_target:
            break
            
        if label == 1:
            if bonafide_count >= bonafide_target:
                continue
            scores, _ = detector.get_score_and_embedding(audio)
            bonafide_scores.append(scores[0])
            bonafide_count += 1
        else:
            if spoof_count >= spoof_target:
                continue
            spoof_count += 1
            
            # Baseline score for the spoof sample
            scores, _ = detector.get_score_and_embedding(audio)
            baseline_score = scores[0]
            baseline_spoof_scores.append(baseline_score)
            
            if agent is not None:
                # 1. Rollout with the agent to find optimized parameters
                env.current_audio = audio
                env.current_step = 0
                _, embeddings = detector.get_score_and_embedding(audio)
                obs = env._get_obs(embeddings[0])
                
                terminated = False
                truncated = False
                last_info = {'score': baseline_score, 'dsp_params': {
                    'jitter': 0, 'shimmer': 0, 'tilt': 0, 'harmonics': 0, 
                    'threshold': -40, 'ratio': 1, 'bitrate': 128000
                }}
                
                while not (terminated or truncated):
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    last_info = info
                
                # 2. Explicitly apply the final DSP configuration and evaluate
                # This ensures we are testing the EXACT final waveform the agent produced
                final_params = last_info['dsp_params']
                adv_audio = env.dsp(audio, final_params)
                
                # Final evaluation of the modified waveform
                adv_scores, _ = detector.get_score_and_embedding(adv_audio)
                final_score = float(adv_scores[0])
                
                adversarial_spoof_scores.append(final_score)
                if final_score > 0.5:
                    success_count += 1
            else:
                adversarial_spoof_scores.append(baseline_score)

        processed_total = bonafide_count + spoof_count
        if processed_total % 10 == 0:
            logger.info(f"Progress: {processed_total}/{num_samples} (B: {bonafide_count}, S: {spoof_count})")

    # 6. Compute Metrics
    logger.info("Computing metrics...")
    
    if len(bonafide_scores) == 0 or spoof_count == 0:
        logger.error(f"Missing classes in evaluated samples (Bonafide: {len(bonafide_scores)}, Spoof: {spoof_count}). EER cannot be computed.")
        return

    bonafide_scores = np.array(bonafide_scores)
    baseline_spoof_scores = np.array(baseline_spoof_scores)
    adversarial_spoof_scores = np.array(adversarial_spoof_scores)
    
    # Baseline Metrics
    base_eer, base_fpr, base_fnr, base_thresholds = compute_eer(bonafide_scores, baseline_spoof_scores)
    base_min_tdcf, _ = compute_mindcf(base_fnr, base_fpr, base_thresholds)
    
    # Adversarial Metrics
    adv_eer, adv_fpr, adv_fnr, adv_thresholds = compute_eer(bonafide_scores, adversarial_spoof_scores)
    adv_min_tdcf, _ = compute_mindcf(adv_fnr, adv_fpr, adv_thresholds)
    
    asr = (success_count / spoof_count) * 100 if spoof_count > 0 else 0
    
    # 7. Report Generation
    total_processed = bonafide_count + spoof_count
    results = {
        "samples": total_processed,
        "bonafide_count": bonafide_count,
        "spoof_count": spoof_count,
        "baseline": {
            "eer": float(base_eer),
            "min_tdcf": float(base_min_tdcf)
        },
        "adversarial": {
            "eer": float(adv_eer),
            "min_tdcf": float(adv_min_tdcf),
            "asr": float(asr)
        },
        "profiling": {
            "batch_size": 4,
            "avg_batch_latency": float(latency),
            "device": str(device)
        }
    }

    with open("outputs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*40)
    print("KON-ARTIST EVALUATION REPORT")
    print("="*40)
    print(f"Samples Evaluated: {total_processed} (Bonafide: {bonafide_count}, Spoof: {spoof_count})")
    print("-" * 40)
    print(f"Baseline EER:      {base_eer*100:.4f}%")
    print(f"Baseline min t-DCF: {base_min_tdcf:.4f}")
    print("-" * 40)
    print(f"Adversarial EER:   {adv_eer*100:.4f}%")
    print(f"Adversarial min t-DCF: {adv_min_tdcf:.4f}")
    print(f"EER Degradation:   {(adv_eer - base_eer)*100:.4f}% (absolute)")
    print("-" * 40)
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    print(f"Inference Latency: {latency/4*1000:.2f}ms / sample")
    print("="*40)
    
    logger.info("Evaluation Complete. Results saved to outputs/evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KON-Artist RL Agent generalization.")
    parser.add_argument("--model", type=str, default="outputs/kon_artist_agent_vectorized", 
                        help="Path to the trained PPO agent.")
    parser.add_argument("--samples", type=int, default=500, 
                        help="Number of samples to evaluate.")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.samples)
