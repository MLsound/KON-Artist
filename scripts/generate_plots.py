"""
Convergence Plot Generator for Reinforcement Learning Training.

This script processes CSV logs containing training rewards and timesteps to 
visualize agent performance. It produces a high-resolution plot featuring 
raw rewards and a smoothed moving average, specifically styled for 
academic reports in the context of the AASIST3 Audio Spoofing attack research.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger(name=__file__,
                    log_file="outputs/metrics_session.log")

def generate_report_plot(csv_path: str, output_path: str, window_size: int = 50):
    """
    Generates the convergence plot required for the course submission.
    
    The plot includes:
    1. Raw rewards per timestep (alpha-blended).
    2. A Simple Moving Average (SMA).
    """
    if not os.path.exists(csv_path):
        logger.error(f"Data file not found at: {csv_path}")
        return

    logger.info(f"Reading data from {csv_path}...")
    
    id = csv_path.split("/")[-1].split(".")[0].replace("rewards_", "")
    if id is not None:
        logger.info(f"Generating plot for training session: {id}")
        folder = os.path.dirname(output_path)
        filename = f"convergence_{id}.png"
        output_path = f"{folder}/{filename}"
        
    try:
        data = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"The CSV file at {csv_path} is empty or contains no parseable columns.")
        return
    
    if data.empty:
        logger.warning("The CSV file is empty.")
        return

    plt.figure(figsize=(12, 6))
    
    # Raw Data
    plt.plot(data['step'], data['reward'], alpha=0.3, color='#1f77b4', label="Reward (Raw)")
    
    # Moving Average
    if len(data) >= window_size:
        data['smooth'] = data['reward'].rolling(window=window_size).mean()
        plt.plot(data['step'], data['smooth'], color='red', linewidth=2, 
                 label=f"Moving Average ({window_size} ep.)")
    
    # FIUBA report aesthetics
    plt.title("KON-Artist Agent Convergence (AASIST3 Attack)", fontsize=14)
    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Reward (Bonafide Probability)", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure the output directory exists
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() # Prevent memory leaks and GUI popups
    logger.info(f"Plot saved successfully at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL convergence plot generator.")
    parser.add_argument("--csv", type=str, default="outputs/history/rewards.csv", 
                        help="Path to the input CSV file.")
    parser.add_argument("--out", type=str, default="outputs/convergence.png", 
                        help="Output path for the image.")
    parser.add_argument("--window", type=int, default=50, 
                        help="Window size for the moving average.")
    
    args = parser.parse_args()
    generate_report_plot(args.csv, args.out, args.window)

# USAGE:
# python -m scripts.generate_plots --csv outputs/history/rewards.csv --out outputs/convergence.png --window 100