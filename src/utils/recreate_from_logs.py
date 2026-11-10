"""
This script extracts step numbers and rewards from training log files and saves them to a CSV file for analysis.
It uses regular expressions to parse lines containing the pattern "Step X/Y ... Reward: Z"

Usage:
    python src/utils/recreate_from_logs.py  -f path/to/logfile.log -o output_rewards.csv
"""

import pandas as pd
import re
import argparse

import ast
import os

def main():
    parser = argparse.ArgumentParser(description="Extract steps, rewards, and DSP parameters from training log files")
    parser.add_argument("-f", "--file", required=True, help="Input log file path")
    parser.add_argument("-o", "--output", default="recreated_rewards.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return

    # Pattern matches Step X | DSP Params: { ... } | Reward: Y
    pattern = re.compile(r"Step\s+(\d+)\s*\|\s*DSP Params:\s*({.*?})\s*\|\s*Reward:\s*([\d.e+-]+)")
    
    data = []
    global_step = 1
    
    with open(args.file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                try:
                    dsp_dict = ast.literal_eval(match.group(2))
                    reward_val = float(match.group(3))
                    
                    row = {
                        "step": global_step,
                        "reward": reward_val,
                        "score": reward_val,  # In this version, score is equivalent to the reward
                    }
                    
                    # Prefix keys with dsp_ to match expected format
                    for k, v in dsp_dict.items():
                        row[f"dsp_{k}"] = v
                        
                    data.append(row)
                    global_step += 1
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}. Error: {e}")
    
    if not data:
        print("No matching log records found.")
        return

    df = pd.DataFrame(data)
    
    # Ensure the output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    df.to_csv(args.output, index=False)
    print(f"Extracted {len(data)} records to {args.output}")

if __name__ == "__main__":
    main()