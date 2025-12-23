import pandas as pd
import networkx as nx
import os

# Constants
INPUT_FILE = 'buildings/990/processed_rssi_2018_03.csv'
BASE_OUTPUT_DIR = 'buildings/990/graphs'
SNAPSHOT_DIR = os.path.join(BASE_OUTPUT_DIR, 'snapshots_2018_03')

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    # Create output directory
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Get all unique time steps
    time_steps = sorted(df['time_step'].unique())
    print(f"Found {len(time_steps)} unique time steps.")

    print("Generating snapshots...")
    
    for t in time_steps:
        # Filter data for this time step
        step_df = df[df['time_step'] == t]
        
        if step_df.empty:
            continue
            
        # Initialize Directed Graph (Snapshot)
        G = nx.DiGraph()
        
        for _, row in step_df.iterrows():
            client = str(row['mac_cliente'])
            ap = str(row['mac_ap'])
            rssi = int(row['rssi'])
            
            # Nodes
            if not G.has_node(client):
                G.add_node(client, type='Client')
            if not G.has_node(ap):
                G.add_node(ap, type='AP')
                
            # Weight: Positive shift for Gephi visualization
            weight = 120 + rssi
            
            # Corrected Direction: AP -> Client (Downlink)
            G.add_edge(ap, client, rssi=rssi, weight=weight)
            
        # Output filename
        filename = f"graph_t_{t}.graphml"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        
        nx.write_graphml(G, filepath)
        
    print(f"Generated {len(time_steps)} snapshot files in {SNAPSHOT_DIR}.")
    print("Done.")

if __name__ == "__main__":
    main()
