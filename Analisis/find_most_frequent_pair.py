import pandas as pd
import glob
import os

def main():
    # Use absolute path for reliability
    base_dir = "/Users/mauriciovieirarodriguez/project/NetROML/buildings/990"
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {base_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Processing...")
    
    counts_list = []
    
    for file in csv_files:
        try:
            # Read only necessary columns
            df = pd.read_csv(file, usecols=['mac_cliente', 'mac_ap'])
            
            # Count pairs in this file
            file_counts = df.groupby(['mac_cliente', 'mac_ap']).size()
            counts_list.append(file_counts)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(file)}: {e}")

    if not counts_list:
        print("No valid data found.")
        return

    # Aggregate all counts
    all_counts = pd.concat(counts_list)
    total_counts = all_counts.groupby(level=[0, 1]).sum()

    if total_counts.empty:
        print("Total counts empty.")
        return

    # Find the maximum
    most_frequent_pair = total_counts.idxmax()
    max_count = total_counts.max()
    
    print("\n" + "="*40)
    print("RESULTADO FINAL")
    print("="*40)
    print(f"La pareja (mac_cliente, mac_ap) más frecuente es:")
    print(f"Cliente: {most_frequent_pair[0]}")
    print(f"AP:      {most_frequent_pair[1]}")
    print(f"Cantidad de apariciones: {int(max_count)}")
    print("="*40)

if __name__ == "__main__":
    main()
