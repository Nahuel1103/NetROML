import pandas as pd
import glob
import os
from pathlib import Path

def merge_and_reindex(input_dir, output_file, file_pattern="rssi_2018_*.csv"):
    """
    Merge all monthly RSSI CSV files in input_dir, drop existing index columns,
    sort, and recreate client_index and block_index.
    """
    input_path = Path(input_dir)
    
    # Find all RSSI CSV files
    csv_pattern = str(input_path / file_pattern)
    csv_files = sorted(glob.glob(csv_pattern))
    
    # Exclude the merged output file if it's in the same directory and matches pattern (though pattern usually won't match "merged")
    csv_files = [f for f in csv_files if os.path.basename(f) != os.path.basename(output_file)]
    
    print(f"Found {len(csv_files)} monthly CSV files in {input_dir}:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
        
    if not csv_files:
        print("No files found to merge.")
        return False
    
    # Load all CSV files
    dfs = []
    for csv_file in csv_files:
        print(f"Loading {os.path.basename(csv_file)}...")
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        return False

    # Concatenate all dataframes
    print("\nMerging all dataframes...")
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merge: {len(merged_df)}")
    
    # Drop client_index and block_index columns if they exist
    columns_to_drop = [col for col in ['client_index', 'block_index'] if col in merged_df.columns]
    if columns_to_drop:
        print(f"Dropping existing index columns: {columns_to_drop}")
        merged_df = merged_df.drop(columns=columns_to_drop)
    
    # Sort by mac_cliente, date, and time
    print("Sorting by mac_cliente, date, and time...")
    merged_df = merged_df.sort_values(by=['mac_cliente', 'date', 'time'], ignore_index=True)
    
    # Create client_index: sequential number for each unique mac_cliente
    print("Creating client_index...")
    merged_df['client_index'] = merged_df.groupby('mac_cliente').ngroup() + 1
    
    # Create block_index: sequential number for each (date, time) block within each mac_cliente
    print("Creating block_index...")
    # This initial group by date/time gives us unique blocks, but we need them sequential per client
    
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
    
    # Reorder columns
    # Ensure all expected columns are present, fill with None if missing (though they should be there)
    expected_cols = ['mac_cliente', 'mac_ap', 'banda', 'antena', 'rssi', 'date', 'time', 'client_index', 'block_index']
    for col in expected_cols:
        if col not in merged_df.columns:
            print(f"Warning: Missing column {col}, adding as empty.")
            merged_df[col] = None
            
    merged_df = merged_df[expected_cols]
    
    # Save to output file
    print(f"\nSaving merged data to {output_file}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    merged_df.to_csv(output_file, index=False)
    
    print(f"✓ Successfully created {output_file}")
    print(f"  Total rows: {len(merged_df)}")
    print(f"  Total unique clients: {merged_df['mac_cliente'].nunique()}")
    print(f"  Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    return True

def main():
    # Default behavior: run on current directory
    script_dir = Path(__file__).parent
    output_file = script_dir / "rssi_merged_all_months.csv"
    merge_and_reindex(script_dir, output_file)

if __name__ == "__main__":
    main()
