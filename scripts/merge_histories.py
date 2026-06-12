import pandas as pd
import argparse
import os

def merge_histories(file1, file2, output_file):
    if not os.path.exists(file1):
        print(f"Error: {file1} not found.")
        return
    if not os.path.exists(file2):
        print(f"Error: {file2} not found.")
        return

    print(f"Reading {file1}...")
    df1 = pd.read_csv(file1)
    print(f"Reading {file2}...")
    df2 = pd.read_csv(file2)
    
    if 'step' not in df1.columns or 'step' not in df2.columns:
        print("Error: Both files must have a 'step' column.")
        return

    last_step_df1 = df1['step'].iloc[-1]
    first_step_df2 = df2['step'].iloc[0]
    
    offset = last_step_df1 + 1 - first_step_df2
    
    print(f"File 1 ends at step {last_step_df1}")
    print(f"File 2 starts at step {first_step_df2}")
    print(f"Adjusting File 2 steps by adding offset: {offset}")
    
    df2['step'] = df2['step'] + offset
    
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # Ensure steps are unique and sorted just in case
    # (Though concat with offset should already handle it if inputs are sequential)
    
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully merged {len(df1)} and {len(df2)} rows.")
    print(f"Merged history saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge two training history CSV files with step adjustment.')
    parser.add_argument('file1', help='First history file (chronologically earlier)')
    parser.add_argument('file2', help='Second history file (continuation)')
    parser.add_argument('--output', '-o', default='rewards_merged.csv', help='Output merged CSV file')
    
    args = parser.parse_args()
    merge_histories(args.file1, args.file2, args.output)
