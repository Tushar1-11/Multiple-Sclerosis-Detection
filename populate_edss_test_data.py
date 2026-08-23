"""
Populate the EDSS column in data/test_data.csv with synthetic values.
"""
import pandas as pd
import numpy as np
import pathlib

TEST_DATA_PATH = pathlib.Path('data/test_data.csv')
OUTPUT_PATH = TEST_DATA_PATH  # overwrite

# Load
df = pd.read_csv(TEST_DATA_PATH, sep='\t')
# Ensure column exists (it may be missing or empty)
if 'EDSS' not in df.columns:
    # Insert after 'Disease duration' if present
    cols = list(df.columns)
    # Find index of 'Disease duration'
    try:
        idx = cols.index('Disease duration')
    except ValueError:
        idx = len(cols) - 1
    cols.insert(idx + 1, 'EDSS')
    df = df.reindex(columns=cols)

# Populate with random values between 0.0 and 7.0 (one decimal)
rng = np.random.default_rng(seed=42)
# Generate values
edss_vals = rng.uniform(0.0, 7.0, size=len(df))
edss_vals = np.round(edss_vals, 1)
df['EDSS'] = edss_vals

# Save back with tab separator
df.to_csv(OUTPUT_PATH, sep='\t', index=False)
print(f"Populated EDSS column for {len(df)} rows in {OUTPUT_PATH}")
