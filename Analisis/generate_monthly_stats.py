import pandas as pd
import os
import glob

def generate_stats():
    # Define paths
    # Define paths based on script location to ensure they are absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # input_dir is ../buildings/990 relative to this script
    input_dir = os.path.abspath(os.path.join(script_dir, "../buildings/990"))
    output_dir = script_dir
    
    # Ensure output directory exists (though it likely does)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of RSSI CSV files
    files = glob.glob(os.path.join(input_dir, "rssi_*.csv"))
    
    if not files:
        print("No matches found for rssi_*.csv in", input_dir)
        return

    print(f"Found {len(files)} files to process.")

    for f in files:
        try:
            print(f"Processing {os.path.basename(f)}...")
            df = pd.read_csv(f)
            
            # Ensure required columns exist
            required_cols = ['mac_cliente', 'date', 'time']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {f}: Missing required columns.")
                continue
                
            # Extract hour from time (format HH:MM)
            # We assume the time format is consistent. If not, pd.to_datetime could be safer but slower.
            # Using string splitting for speed and simplicity given the viewed data format which is clean.
            df['hour'] = df['time'].astype(str).apply(lambda x: x.split(':')[0] if ':' in x else x)
            
            # Deduplicate to count 1 for each unique (mac_cliente, date, hour) tuple
            # "una mac_cliente para un día y hora puede tener más de un dato pero debe contarse como frecuencia 1"
            unique_slots = df[['mac_cliente', 'date', 'hour']].drop_duplicates()
            
            # Calculate frequency: count of unique slots per client
            freq = unique_slots['mac_cliente'].value_counts().reset_index()
            freq.columns = ['mac_cliente', 'frecuencia']
            
            # Save to output file
            base_name = os.path.basename(f)
            output_filename = f"stats_{base_name}"
            output_path = os.path.join(output_dir, output_filename)
            
            freq.to_csv(output_path, index=False)
            print(f"Saved {output_path}")
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    generate_stats()
