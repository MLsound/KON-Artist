"""
Unified Session Analysis Orchestrator.

This script automates the full post-training analysis flow:
1. Extracts 'winners' (samples with score >= 0.5).
2. Performs cluster-based differentiation analysis.
3. Exports both winners and analysis summaries.

Usage:
    python -m scripts.analyze_session -f outputs/history/rewards_20260603_203248.csv -t 0.0
"""

import argparse
import os
import sys
import pandas as pd
from src.utils.analysis import get_winners, analyze_cluster_differentiation, extract_id_from_filename
from src.utils.logger import get_logger
import logging

logger = get_logger(name=__file__, level=logging.INFO)

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
    logger.info(f"Analyzing session {file_id} from {args.file}...")
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
        winners_df.to_csv(winners_path, index=False)
        logger.info(f"Saved {len(winners_df)} winners to {winners_path}")
        
        # 3. Analyze Clusters
        logger.info("Performing cluster differentiation analysis...")
        summary_df = analyze_cluster_differentiation(winners_df)
        
        if summary_df.empty:
            logger.warning("Could not perform cluster analysis (missing 'cluster' column or data).")
        else:
            # Calculate ASR per cluster using the full dataframe df
            if 'cluster' not in df.columns and 'cluster_id' in df.columns:
                df = df.rename(columns={'cluster_id': 'cluster'})
            total_counts = df.groupby('cluster').size()
            
            # Divide count of winners in cluster by total count of samples in cluster, times 100
            summary_df['asr'] = (summary_df['count'] / total_counts * 100).fillna(0)
            
            # Reorder columns to have count first, then asr, then metrics, then dsp_cols
            cols = list(summary_df.columns)
            if 'asr' in cols:
                cols.remove('asr')
                cols.insert(1, 'asr')
                summary_df = summary_df[cols]
                
            summary_path = os.path.join(args.out_dir, f"analysis_clusters{file_id}{thresh_ref}.csv")
            summary_df.to_csv(summary_path)
            
            # Console output for immediate feedback
            print("\n" + "="*80)
            print(f"CLUSTER DIFFERENTIATION SUMMARY (Session {file_id})")
            print("="*80)
            print(summary_df.to_string())
            print("="*80)
            logger.info(f"Analysis summary saved to {summary_path}")

            # 4. Generate Plots
            try:
                import matplotlib.pyplot as plt
                
                # --- Plot 1: Per-Cluster Attack Success Rate (ASR) ---
                plt.figure(figsize=(10, 6), dpi=300)
                ax1 = summary_df['asr'].plot(kind='bar', color='#66bb6a', alpha=0.9, width=0.6)
                plt.title("Per-Cluster Attack Success Rate (ASR)", fontsize=14, pad=15)
                plt.xlabel("Acoustic Cluster ID", fontsize=12, labelpad=10)
                plt.ylabel("ASR (%)", fontsize=12, labelpad=10)
                plt.ylim(0, 105)
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                
                # Annotate ASR percentages on top of bars
                for i, p in enumerate(ax1.patches):
                    cluster_id = summary_df.index[i]
                    asr_val = summary_df.loc[cluster_id, 'asr']
                    ax1.annotate(f"{asr_val:.1f}%", 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='bottom', xytext=(0, 3), 
                                 textcoords='offset points', fontsize=10)
                
                asr_plot_path = os.path.join(args.out_dir, f"cluster_vulnerability{file_id}{thresh_ref}.png")
                plt.tight_layout()
                plt.savefig(asr_plot_path, dpi=300)
                plt.close()
                logger.info(f"Cluster vulnerability (ASR) plot saved to {asr_plot_path}")

                # --- Plot 2: Per-Cluster Average Deception Score ---
                plt.figure(figsize=(10, 6), dpi=300)
                ax2 = summary_df['score'].plot(kind='bar', color='#2196f3', alpha=0.9, width=0.6)
                plt.title("Per-Cluster Average Deception Score", fontsize=14, pad=15)
                plt.xlabel("Acoustic Cluster ID", fontsize=12, labelpad=10)
                plt.ylabel("Average Deception Score (Vulnerability)", fontsize=12, labelpad=10)
                plt.ylim(0, 1.05)
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                
                # Annotate deception scores on top of bars
                for i, p in enumerate(ax2.patches):
                    cluster_id = summary_df.index[i]
                    score_val = summary_df.loc[cluster_id, 'score']
                    ax2.annotate(f"{score_val:.3f}", 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='bottom', xytext=(0, 3), 
                                 textcoords='offset points', fontsize=10)
                
                score_plot_path = os.path.join(args.out_dir, f"cluster_vulnerability_score{file_id}{thresh_ref}.png")
                plt.tight_layout()
                plt.savefig(score_plot_path, dpi=300)
                plt.close()
                logger.info(f"Cluster vulnerability (Score) plot saved to {score_plot_path}")

            except Exception as plot_err:
                logger.error(f"Failed to generate plots: {plot_err}")

if __name__ == "__main__":
    main()

# Usage: python -m scripts.analyze_session -f path/to/rewards.csv