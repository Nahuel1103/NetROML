import os
import glob
import client_analysis as ca

def reorder_all_files():
    buildings_dir = "buildings"
    file_pattern = os.path.join(buildings_dir, "**", "rssi_*.csv")
    files = glob.glob(file_pattern, recursive=True)
    
    print(f"Found {len(files)} files to reorder.")
    
    for filepath in sorted(files):
        print(f"Processing {filepath}...")
        try:
            # Load
            df = ca.load_client_data(filepath)
            
            # Sort
            df_sorted = ca.sort_records_by_connection(df)
            
            # Save (overwrite)
            df_sorted.to_csv(filepath, index=False)
            print(f"  Done.")
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

if __name__ == "__main__":
    reorder_all_files()
