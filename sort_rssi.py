import pandas as pd
import glob
import os

def sort_rssi_files():
    # Define the directory
    directory = '/Users/mauriciovieirarodriguez/project/NetROML/buildings/990'
    
    # Find all rssi_*.csv files matched
    files = glob.glob(os.path.join(directory, 'rssi_*.csv'))
    
    if not files:
        print(f"No files found in {directory}")
        return

    print(f"Found {len(files)} files to process.")

    for file_path in files:
        try:
            print(f"Processing {os.path.basename(file_path)}...")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            required_columns = ['mac_cliente', 'time', 'date']
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping {os.path.basename(file_path)}: Missing one or more required columns ({required_columns})")
                continue
                
            # Sort the DataFrame
            df_sorted = df.sort_values(by=['mac_cliente', 'time', 'date'])
            
            # Create client_index
            unique_clients = sorted(df_sorted['mac_cliente'].unique())
            client_map = {client: i+1 for i, client in enumerate(unique_clients)}
            df_sorted['client_index'] = df_sorted['mac_cliente'].map(client_map)
            
            # Create block_index
            # Group by client, then assign sequential ID to unique (time, date) tuples
            # ngroup() gives 0-based index for each group (here, each time block within a client)
            df_sorted['block_index'] = df_sorted.groupby(['mac_cliente', 'time', 'date'], sort=False).ngroup()
            
            # The above ngroup is global across the groupby if we are not careful. 
            # We want it to be per client, starting at 1.
            # A cleaner way: group by client, then apply a transform or a custom function.
            
            def assign_block_index(group):
                # Calculate unique time blocks and assign codes
                # Method: create a 'time_block' from date+time, factorize it (or ngroup), then add 1
                # Since the group is already sorted by date+time (from df_sorted), factorize should work
                # or simpler: just use ngroup on the subgroup if possible, but pandas groupby on subgroup is tricky in apply
                
                # Using factorize on the tuples of (date, time)
                # But factorize doesn't guarantee order unless sorted. We are sorted.
                times = group['time'].astype(str) + '_' + group['date'].astype(str)
                 # factorize returns (codes, uniques). codes are 0-based index of unique values.
                 # if 'sort=True' is not guaranteed to respect occurrence order in older pandas versions 
                 # for factorize, actually pd.factorize(sort=True) sorts unique values.
                 # We want the order of appearance which corresponds to time since we sorted.
                codes, _ = pd.factorize(times, sort=False) 
                return codes + 1

            # However, pd.factorize might not produce strictly sequential increasing numbers if the data wasn't sorted perfectly by time.
            # But we did sort by ['mac_cliente', 'date', 'time'].
            
            # Let's use a simpler approach that is robust:
            # For each client, we want sequential block numbers.
            # rank(method='dense') on the (date,time) column(s)?
            
            df_sorted['block_index'] = df_sorted.groupby('mac_cliente')[['time', 'date']].apply(
                lambda x: x.groupby(['time', 'date'], sort=False).ngroup() + 1
            ).reset_index(level=0, drop=True) 
            # Note: The groupby inside apply needs to verify 'ngroup' behavior or use 'dense' rank
            
            # Alternative:
            # 1. dropped duplicates of (mac_cliente, date, time) to get unique blocks
            # 2. assign rank/counter
            # 3. merge back
            
            # Let's try the groupby + ngroup equivalent but correctly.
            # dense_rank of (date, time) within mac_cliente
            
            # To ensure it works as expected:
            # Sort is guaranteed.
            # df_sorted['block_index'] = df_sorted.groupby('mac_cliente').apply(lambda x: x.set_index(['date', 'time']).index.factorize()[0] + 1).reset_index(level=0, drop=True)
            # This might be tricky with index alignment if duplicates exist.

            # Safest pandas way usually:
            # df_sorted['temp_id'] = df_sorted['date'].astype(str) + df_sorted['time'].astype(str)
            # df_sorted['block_index'] = df_sorted.groupby('mac_cliente')['temp_id'].rank(method='dense').astype(int)
            
            # Let's verify 'rank' behaviour with strings. It ranks alphabetically.
            # Since date is YYYY-MM-DD and time is HH:MM, alphabetical order IS chronological order.
            # So rank(method='dense') should work perfectly.
            
            df_sorted['timestamp_str'] = df_sorted['time'].astype(str) + ' ' + df_sorted['date'].astype(str)
            df_sorted['block_index'] = df_sorted.groupby('mac_cliente')['timestamp_str'].rank(method='dense').astype(int)
            df_sorted.drop(columns=['timestamp_str'], inplace=True)
            
            # Save the sorted DataFrame back to the same file
            df_sorted.to_csv(file_path, index=False)
            
            print(f"Successfully sorted and updated {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    sort_rssi_files()
