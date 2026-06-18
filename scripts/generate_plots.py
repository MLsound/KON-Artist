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
                    log_file="outputs/logs/metrics_session.log")

def generate_report_plot(csv_path: str, output_path: str, window_size: int = 50, limit: int = 0, dots: bool = False):
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
        filename = f"convergence_{id}.png" if not dots else f"convergence_{id}_dots.png"
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
    args = {'alpha': 0.3, 'color': '#1f77b4', 'marker': '.', 'linestyle': '', 'linewidth': 0, 'label': "Reward (Raw)"} if dots else {'alpha': 0.8, 'color': '#1f77b4', 'linewidth': 0.2, 'label': "Reward (Raw)"}
    plt.plot(data['step'], data['reward'], **args)
    if dots: print(f"Plotted raw rewards with {len(data)} data points in dots mode.")

    # Moving Average
    if len(data) >= window_size:
        slow_ma_rate = 0.1 # Ratio for the second moving average window size
        slow_window = int(window_size * slow_ma_rate) # Ensure the window isn't larger than half the data length
        data['smooth2'] = data['reward'].rolling(window=slow_window).mean()
        plt.plot(data['step'], data['smooth2'], color='orange', linewidth=1, 
                 label=f"Faster MA ({slow_window} ep.)")
        data['smooth'] = data['reward'].rolling(window=window_size).mean()
        plt.plot(data['step'], data['smooth'], color='red', linewidth=2, 
                 label=f"Slower MA ({window_size} ep.)")
        
    # Limit y-axis to avoid outlier skew and ensure 0 is the bottom
    if limit > 0:
        plt.ylim(0, limit)
        logger.info(f"Y-axis limited to: (0, {limit})")
    else:
        plt.ylim(bottom=0)
    
    # FIUBA report aesthetics
    plt.title("KON-Artist Agent Convergence (AASIST3 Attack)", fontsize=14)
    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Reward (Bonafide Probability)", fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a text box with statistics summary
    try:
        final_reward = data['reward'].iloc[-1]
        mean_last_100 = data['reward'].tail(100).mean()
        max_reward = data['reward'].max()
        
        stats_text = (f"Final Reward: {final_reward:.3f}\n"
                      f"Mean (last 100): {mean_last_100:.3f}\n"
                      f"Max Reward: {max_reward:.3f}")
        
        plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                       verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='#1f77b4'))
    except Exception as e:
        logger.error(f"Failed to calculate stats for text box: {e}")
    
    # Ensure the output directory exists
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() # Prevent memory leaks and GUI popups
    logger.info(f"Plot saved successfully at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL convergence plot generator.")
    parser.add_argument("--csv", type=str, default="outputs/history/rewards.csv", 
                        help="Path to the input CSV file.")
    parser.add_argument("--out", type=str, default="outputs/convergence.png", 
                        help="Output path for the image.")
    parser.add_argument("--window", type=int, default=1000, 
                        help="Window size for the moving average.")
    parser.add_argument("--limit", "-l", type=int, default=0, 
                        help="Set a fixed higher limit for the y-axis (0 to disable).")
    parser.add_argument("--dots", "-d", action="store_true", 
                        help="Plot raw rewards as dots instead of lines.")
    
    args = parser.parse_args()
    generate_report_plot(csv_path=args.csv,
                         output_path=args.out,
                         window_size=args.window,
                         limit=args.limit,
                         dots=args.dots
                         )

# USAGE:
# python -m scripts.generate_plots --csv outputs/history/rewards.csv --out outputs/convergence.png --window 100