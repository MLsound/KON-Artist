import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', required=True, help='Input CSV filename')
args = parser.parse_args()

f=pd.read_csv(args.filename)
s=f.sort_values(by="bonus",ascending=False)
s=s[s['score']>=0.50]

# Extract ID from input filename (pattern: _YYYYMMDD_HHMMSS)
match = re.search(r'_\d{8}_\d{6}', args.filename)
file_id = match.group(0) if match else ''
output_filename = f'winners{file_id}.csv'

s.to_csv(output_filename)

# Usage: python get_winners.py -f path/to/your/input.csv