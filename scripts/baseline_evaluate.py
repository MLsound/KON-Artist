"""
Baseline AASIST3 Performance Evaluation Script.

This script executes a full evaluation cycle of the AASIST3 detector 
against a specified dataset using local protocol files. It computes 
standard performance metrics, including the Equal Error Rate (EER) 
and Minimum Detection Cost Function (min t-DCF).
"""

import argparse
import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from src.models.aasist import AASISTWrapper
from src.utils.metrics import compute_eer, compute_mindcf
from src.utils.logger import get_logger
from src.utils.audio import preprocess_audio
import logging

# Initialize logger
logger = get_logger(name=__file__,
                    log_file="outputs/logs/baseline_eval.log",
                    level=logging.INFO)

def evaluate(model_path, data_dir, protocol_path, scores_out=None, device="cuda"):
    """
    Evaluates AASIST3 detector performance against a local dataset.
    """
    # 1. Initialize the detector wrapper
    logger.info(f"Loading AASIST3 model from {model_path} on {device}...")
    detector = AASISTWrapper(model_path, device=device)
    
    bonafide_scores = []
    spoof_scores = []
    results_to_save = []
    
    # 2. Read protocol (standard format: [speaker, file_id, -, tag, label])
    if not os.path.exists(protocol_path):
        logger.error(f"Protocol file not found: {protocol_path}")
        return
        
    logger.info(f"Loading protocol from: {protocol_path}")
    with open(protocol_path, 'r') as f:
        lines = f.readlines()

    logger.info(f"Starting evaluation of {len(lines)} files...")
    
    # 3. Inference loop
    for line in tqdm(lines, desc="Evaluating"):
        parts = line.strip().split()
        if len(parts) < 5: 
            continue
        
        file_id = parts[1]
        label = parts[4] # 'bonafide' or 'spoof'
        
        # Try both .wav and other potential extensions if needed
        file_path = os.path.join(data_dir, f"{file_id}.wav")
        
        if not os.path.exists(file_path):
            logger.debug(f"File missing: {file_path}")
            continue
            
        try:
            # Load audio and retrieve scores from the wrapper
            waveform, sr = torchaudio.load(file_path)
            
            # Apply standard preprocessing
            processed_audio = preprocess_audio(waveform, sr)
            
            # AASISTWrapper.get_score_and_embedding handles batch/squeezing
            scores, _ = detector.get_score_and_embedding(processed_audio)
            score = float(scores[0])
            
            if label == 'bonafide':
                bonafide_scores.append(score)
            else:
                spoof_scores.append(score)
            
            # Format results for text output
            results_to_save.append(f"{file_id} {score} {label}\n")
            
        except Exception as e:
            logger.error(f"Error processing {file_id}: {str(e)}")

    # 4. Save individual scores for further analysis
    if scores_out and results_to_save:
        os.makedirs(os.path.dirname(scores_out), exist_ok=True)
        with open(scores_out, 'w') as f:
            f.writelines(results_to_save)
        logger.info(f"Individual scores saved at: {scores_out}")

    # 5. Metrics calculation
    if not bonafide_scores or not spoof_scores:
        logger.warning("Insufficient score data to calculate metrics.")
        return None

    logger.info("Computing final metrics...")
    
    bonafide_scores = np.array(bonafide_scores)
    spoof_scores = np.array(spoof_scores)
    
    # Equal Error Rate
    eer, fpr, fnr, thresholds = compute_eer(bonafide_scores, spoof_scores)
    
    # min t-DCF (using default ASVspoof costs)
    min_tdcf, _ = compute_mindcf(fnr, fpr, thresholds)
    
    # Console output
    print("\n" + "="*40)
    print("AASIST3 BASELINE EVALUATION RESULTS")
    print("="*40)
    print(f"Dataset:          {os.path.basename(data_dir)}")
    print(f"Bonafide Samples: {len(bonafide_scores)}")
    print(f"Spoof Samples:    {len(spoof_scores)}")
    print("-" * 40)
    print(f"EER:              {eer*100:.4f}%")
    print(f"min t-DCF:        {min_tdcf:.4f}")
    print("="*40 + "\n")
    
    logger.info(f"Evaluation Complete. EER: {eer*100:.2f}%, min-tDCF: {min_tdcf:.4f}")
    return eer, min_tdcf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline AASIST3 Evaluation Script.")
    parser.add_argument("--model", type=str, default="MTUCI/AASIST3", 
                        help="Path to the model weight (.pth) or HuggingFace ID.")
    parser.add_argument("--data", type=str, required=True, 
                        help="Directory containing .wav files.")
    parser.add_argument("--protocol", type=str, required=True, 
                        help="Path to the protocol/label file.")
    parser.add_argument("--save-scores", type=str, default="outputs/baseline/baseline_scores.txt", 
                        help="Path to save individual inference scores.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    args = parser.parse_args()
    evaluate(args.model, args.data, args.protocol, args.save_scores, args.device)

# USAGE EXAMPLE:
# python -m scripts.baseline_evaluate --data data/eval/wav --protocol data/eval/protocol.txt
