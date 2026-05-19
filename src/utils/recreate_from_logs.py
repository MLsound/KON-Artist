import pandas as pd
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract steps and rewards from training log files")
    parser.add_argument("-f", "--file", required=True, help="Input log file path")
    parser.add_argument("-o", "--output", default="recreated_rewards.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Compiling the regex pattern beforehand improves performance when looping through many lines
    # \d+ matches digits, \s+ matches whitespace, and ( ) captures the targeted groups
    pattern = re.compile(r"Step\s+(\d+)/\d+.*?Reward:\s+([\d.-]+)")
    
    steps, rewards = [], []
    
    with open(args.file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # group(1) corresponds to the step, group(2) corresponds to the reward
                steps.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
    
    # Save the extracted data to a CSV file for easier analysis
    df = pd.DataFrame({"step": steps, "reward": rewards})
    df.to_csv(args.output, index=False)
    print(f"Extracted {len(steps)} records to {args.output}")

if __name__ == "__main__":
    main()