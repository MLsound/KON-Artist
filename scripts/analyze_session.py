"""
Unified Session Analysis Orchestrator.

This script automates the full post-training analysis flow:
1. Extracts 'winners' (samples with score >= threshold).
2. Computes session-wide DSP parameter statistics (mean, min, max).
3. Exports both winners and analysis summaries.

Usage:
    python -m scripts.analyze_session -f outputs/history/rewards_20260603_203248.csv -t 0.5
"""

import argparse
import os
import re
import sys
import pandas as pd
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__, level=logging.INFO)


def extract_id_from_filename(filename):
    """
    Extracts the timestamp identifier from a file name.
    """
    match = re.search(r"\d{8}_\d{6}", os.path.basename(filename))
    return f"_{match.group(0)}" if match else ""


def get_winners(df, score_threshold=0.5):
    """
    Filters the dataframe to keep only successful model-deception vectors.
    """
    if "score" not in df.columns:
        df.columns = [c.lower() for c in df.columns]
    if "score" in df.columns:
        return df[df["score"] >= score_threshold].copy()
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Full session analysis orchestrator.")
    parser.add_argument('-f', '--file', required=True, help='Path to the rewards CSV file.')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Score threshold for winners.')
    parser.add_argument('--out-dir', default='outputs/analysis/', help='Directory to save results.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    os.makedirs(args.out_dir, exist_ok=True)
    file_id = extract_id_from_filename(args.file)
    thresh_ref = f"_thresh{args.threshold:.2f}" if args.threshold != 0.5 else ""
    
    # 1. Load Data
    logger.info(f"Analyzing session {file_id.strip('_')} from {args.file}...")
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)
        
    # 2. Extract Winners
    logger.info(f"Extracting winners (score >= {args.threshold})...")
    winners_df = get_winners(df, score_threshold=args.threshold)
    
    if winners_df.empty:
        logger.warning("No winners found with the given threshold.")
    else:
        winners_path = os.path.join(args.out_dir, f"winners{file_id}{thresh_ref}.csv")
        # Save sorted winners as in the original workspace code
        if "bonus" in winners_df.columns:
            winners_df = winners_df.sort_values(by="bonus", ascending=False)
        winners_df.to_csv(winners_path, index=False)
        logger.info(f"Saved {len(winners_df)} winners to {winners_path}")
        
        # 3. Analyze DSP Parameters
        logger.info("Performing DSP parameter statistics analysis...")
        winners_df.columns = [c.lower() for c in winners_df.columns]
        dsp_cols = sorted([col for col in winners_df.columns if col.startswith("dsp_")])
        
        if not dsp_cols:
            logger.warning("No DSP parameters (columns starting with 'dsp_') found in the dataset.")
        else:
            summary_data = []
            for col in dsp_cols:
                summary_data.append({
                    "dsp_parameter": col,
                    "mean": winners_df[col].mean(),
                    "min": winners_df[col].min(),
                    "max": winners_df[col].max()
                })
            summary_df = pd.DataFrame(summary_data)
            summary_df.set_index("dsp_parameter", inplace=True)
            
            summary_path = os.path.join(args.out_dir, f"analysis{file_id}{thresh_ref}.csv")
            summary_df.to_csv(summary_path)
            
            # Console output for immediate feedback
            print("\n" + "="*80)
            print(f"SESSION DSP PARAMETERS ANALYSIS (Session {file_id.strip('_')})")
            print("="*80)
            print(summary_df.to_string())
            print("="*80)
            logger.info(f"Analysis summary saved to {summary_path}")


if __name__ == "__main__":
    main()