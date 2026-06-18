"""
Analysis Utilities for KON-Artist.

This module provides functions to process training logs, extract successful 
attack samples (winners), and analyze acoustic cluster differentiation.
"""

import pandas as pd
import re
import os
import logging

def get_winners(df: pd.DataFrame, score_threshold: float = 0.5) -> pd.DataFrame:
    """
    Filters and sorts the dataframe to identify 'winners' (samples that fooled the detector).
    """
    if df.empty:
        return df
    
    # Ensure mandatory columns exist
    if 'score' not in df.columns or 'bonus' not in df.columns:
        return pd.DataFrame()

    winners = df[df['score'] >= score_threshold].copy()
    winners = winners.sort_values(by="bonus", ascending=False)
    return winners

def analyze_cluster_differentiation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups winners by cluster and calculates mean metrics and DSP parameters.
    """
    if df.empty:
        return df

    # Normalize cluster column name
    if 'cluster' not in df.columns and 'cluster_id' in df.columns:
        df = df.rename(columns={'cluster_id': 'cluster'})
    
    if 'cluster' not in df.columns:
        return pd.DataFrame()

    # Identify columns to aggregate
    dsp_cols = [col for col in df.columns if col.startswith('dsp_')]
    metrics = ['score', 'bonus']
    if 'reward' in df.columns:
        metrics.append('reward')
    
    agg_cols = metrics + dsp_cols
    
    # Filter out rows with no cluster information
    df = df.dropna(subset=['cluster'])
    if df.empty:
        return pd.DataFrame()

    # Aggregate
    summary = df.groupby('cluster')[agg_cols].mean()
    summary['count'] = df.groupby('cluster').size()
    
    # Reorder columns: count first
    ordered_cols = ['count'] + metrics + dsp_cols
    summary = summary[ordered_cols]
    
    return summary

def extract_id_from_filename(filename: str) -> str:
    """
    Extracts the timestamp ID from a standard rewards file name.
    """
    match = re.search(r'_\d{8}_\d{6}', filename)
    return match.group(0) if match else ''
