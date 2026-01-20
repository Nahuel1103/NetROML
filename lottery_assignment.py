import pandas as pd
import numpy as np

def assign_lottery(input_file, output_file):
    """
    Reads the RSSI CSV file, assigns random timeslot and duration to each unique block,
    and saves the processed data.
    """
    df = pd.read_csv(input_file)
    
    # Ensure correct data types
    df['client_index'] = df['client_index'].astype(str)
    df['block_index'] = df['block_index'].astype(str) # ensure block index is treated as group identifier
    
    # Identify unique blocks. A block is defined by a unique combination of client_index and block_index
    # However, looking at the data, 'block_index' seems to reset or be sequential per client?
    # Let's group by both just to be safe and unique.
    
    # Create a unique identifier for blocks to assign lottery values
    unique_blocks = df[['client_index', 'block_index']].drop_duplicates()
    
    # Generate random values
    # Timeslots: 0 to 20 (arbitrary range for simulation)
    # Duration: 1 to 5
    unique_blocks['timeslot'] = np.random.randint(0, 21, size=len(unique_blocks))
    unique_blocks['duration'] = np.random.randint(1, 6, size=len(unique_blocks))
    
    # Merge back to original dataframe
    df_processed = pd.merge(df, unique_blocks, on=['client_index', 'block_index'], how='left')
    
    # Save processed file
    df_processed.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    print(df_processed[['client_index', 'block_index', 'timeslot', 'duration']].head(10))

if __name__ == "__main__":
    assign_lottery('rssi_2018_08.csv', 'rssi_2018_08_processed.csv')
