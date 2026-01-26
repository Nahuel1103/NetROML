#!/usr/bin/env python3
"""
Merge all monthly RSSI CSV files, drop existing client_index and block_index columns,
sort by mac_cliente, date, and time, then recreate both index columns.
"""

import pandas as pd
import glob
import os
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Find all RSSI CSV files (excluding any merged output files)
    csv_pattern = str(script_dir / "rssi_2018_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    # Exclude the merged output file if it exists
    csv_files = [f for f in csv_files if 'merged' not in os.path.basename(f)]
    
    print(f"Found {len(csv_files)} monthly CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load all CSV files
    dfs = []
    for csv_file in csv_files:
        print(f"Loading {os.path.basename(csv_file)}...")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # Concatenate all dataframes
    print("\nMerging all dataframes...")
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merge: {len(merged_df)}")
    
    # Drop client_index and block_index columns if they exist
    columns_to_drop = [col for col in ['client_index', 'block_index'] if col in merged_df.columns]
    if columns_to_drop:
        print(f"Dropping columns: {columns_to_drop}")
        merged_df = merged_df.drop(columns=columns_to_drop)
    
    # Sort by mac_cliente, date, and time
    print("Sorting by mac_cliente, date, and time...")
    merged_df = merged_df.sort_values(by=['mac_cliente', 'date', 'time'], ignore_index=True)
    
    # Create client_index: sequential number for each unique mac_cliente
    print("Creating client_index...")
    merged_df['client_index'] = merged_df.groupby('mac_cliente').ngroup() + 1
    
    # Create block_index: sequential number for each (date, time) block within each mac_cliente
    print("Creating block_index...")
    merged_df['block_index'] = merged_df.groupby(['mac_cliente', 'date', 'time']).ngroup()
    
    # For each client, we need to renumber block_index starting from 1
    # First, create a temporary column with the block number within each client
    def calculate_block_index(group):
        # Get unique (date, time) combinations for this client
        unique_blocks = group[['date', 'time']].drop_duplicates()
        # Create a mapping from (date, time) to block_index
        unique_blocks = unique_blocks.sort_values(by=['date', 'time']).reset_index(drop=True)
        unique_blocks['block_idx'] = range(1, len(unique_blocks) + 1)
        # Merge back to get block_index for each row
        result = group.merge(unique_blocks, on=['date', 'time'], how='left')
        return result['block_idx']
    
    merged_df['block_index'] = merged_df.groupby('mac_cliente', group_keys=False).apply(calculate_block_index).values
    
    # Reorder columns to match original format
    column_order = ['mac_cliente', 'mac_ap', 'banda', 'antena', 'rssi', 'date', 'time', 'client_index', 'block_index']
    merged_df = merged_df[column_order]
    
    # Save to output file
    output_file = script_dir / "rssi_merged_all_months.csv"
    print(f"\nSaving merged data to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    print(f"✓ Successfully created {output_file}")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Total unique clients: {merged_df['mac_cliente'].nunique()}")
    print(f"  Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")

if __name__ == "__main__":
    main()
