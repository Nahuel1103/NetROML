import pandas as pd
import numpy as np
import os
import random
import networkx as nx
import pickle
from scipy.interpolate import interp1d

# Constants
INPUT_FILE = 'buildings/990/rssi_2018_03.csv'
OUTPUT_FILE = 'buildings/990/processed_rssi_advanced.csv'
GRAPH_DIR = 'buildings/990/graphs_advanced'
PICKLE_DIR = os.path.join(GRAPH_DIR, 'pkl')
GRAPHML_DIR = os.path.join(GRAPH_DIR, 'graphml')

import shutil

def ensure_dirs():
    for d in [GRAPH_DIR, PICKLE_DIR, GRAPHML_DIR]:
        if os.path.exists(d):
            # Clean up old files
            for f in os.listdir(d):
                file_path = os.path.join(d, f)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(d, exist_ok=True)

def load_data(filepath):
    print(f"Loading {filepath}...")
    return pd.read_csv(filepath)

def filter_frequent_clients(df):
    counts = df['mac_cliente'].value_counts()
    median_val = counts.median()
    frequent_clients = counts[counts >= median_val].index
    print(f"Filtering clients: Median={median_val}, Total={len(counts)}, Frequent={len(frequent_clients)}")
    return df[df['mac_cliente'].isin(frequent_clients)]

def aggregate_highest_rssi(df):
    # For each group of Client, AP, Band, and "Scan", pick the max RSSI.
    # Since we don't have a scan ID, we'll group by (mac_cliente, mac_ap, banda)
    # and then identify contiguous blocks or just take the max if we assume 
    #antenna diversity is the main cause of multiple records.
    # The requirement says: "Selecciono la antena con mayor rssi para cada banda para el par {mac_client, mac_ap}"
    # This might mean for the entire month, or per scan.
    # Usually it means per measurement event.
    # Without timestamps, let's assume records are grouped in scans of ~2-4.
    # Let's try to group by (mac_cliente, mac_ap, banda, antena) and then aggregate 
    # to get one record per "event".
    
    # Actually, a simpler interpretation: for each connection sequence, 
    # we want to collapse antenna diversity.
    # Let's group by (mac_cliente, mac_ap, banda) and take max RSSI for each group
    # if we want a single point, but we want a SEQUENCE.
    
    # Let's assume every N rows for the same (client, ap, band) is a scan.
    # Or better: just keep all rows but for each "instant" (which we don't have) take max.
    # Let's try to group by index // 4 (heuristic for 4 antennas) within each (client, ap, band) group.
    
    print("Selecting highest RSSI per antenna diversity...")
    
    def process_group(g):
        # Heuristic: every 4 rows is a scan (ant0, ant1, etc)
        g = g.reset_index(drop=True)
        g['scan_id'] = g.index // 4 
        # Group by scan_id and banda to take max rssi
        res = g.groupby(['scan_id', 'banda']).agg({
            'mac_cliente': 'first',
            'mac_ap': 'first',
            'rssi': 'max'
        }).reset_index() # Keep scan_id and banda
        return res.drop(columns=['scan_id'])

    df_agg = df.groupby(['mac_cliente', 'mac_ap', 'banda'], sort=False).apply(process_group).reset_index(drop=True)
    return df_agg

def interpolate_sequences(df):
    print("Interpolating connection sequences...")
    processed_rows = []
    
    grouped = df.groupby(['mac_cliente', 'mac_ap'], sort=False)
    
    for (client, ap), group in grouped:
        # User requested random duration
        target_len = random.randint(10, 50)
        
        # We might have multiple bands in the group. Let's process each band independently?
        # "Selecciono la antena con mayor rssi para cada banda para el par {mac_client, mac_ap}"
        # This implies we keep banda info.
        
        for band in group['banda'].unique():
            band_group = group[group['banda'] == band]
            y = band_group['rssi'].values
            if len(y) < 2:
                # Can't interpolate with 1 point, just repeat it
                new_y = np.repeat(y, target_len)
            else:
                x = np.linspace(0, 1, num=len(y))
                f = interp1d(x, y, kind='linear', fill_value="extrapolate")
                new_x = np.linspace(0, 1, num=target_len)
                new_y = f(new_x).astype(int)
            
            processed_rows.append({
                'mac_cliente': client,
                'mac_ap': ap,
                'banda': band,
                'rssi_seq': new_y
            })
            
    return processed_rows

def schedule_timesteps(interpolated_data):
    print("Scheduling timesteps with constraints...")
    # Constraint: mac_client cannot be connected to more than one mac_ap at the same time_step.
    
    # Sort clients to process one by one
    client_data = {}
    for item in interpolated_data:
        client = item['mac_cliente']
        if client not in client_data:
            client_data[client] = []
        client_data[client].append(item)
        
    final_rows = []
    
    for client, connections in client_data.items():
        current_time = 0
        # We can randomize start time for each client to scatter them
        current_time = random.randint(0, 100)
        
        for conn in connections:
            seq = conn['rssi_seq']
            start_t = current_time
            for i, rssi in enumerate(seq):
                final_rows.append({
                    'mac_cliente': client,
                    'mac_ap': conn['mac_ap'],
                    'banda': conn['banda'],
                    'rssi': rssi,
                    'time_step': start_t + i
                })
            # Add a small gap between connections of the same client
            current_time = start_t + len(seq) + random.randint(5, 15)
            
    return pd.DataFrame(final_rows)

def generate_graphs(df):
    print("Generating graphs...")
    time_steps = sorted(df['time_step'].unique())
    
    for t in time_steps:
        step_df = df[df['time_step'] == t]
        
        # Iterate over unique bands in this timestep
        for band in step_df['banda'].unique():
            band_df = step_df[step_df['banda'] == band]
            
            G = nx.DiGraph()
            
            for _, row in band_df.iterrows():
                client = str(row['mac_cliente'])
                ap = str(row['mac_ap'])
                rssi = int(row['rssi'])
                
                # Gephi weight
                weight = 120 + rssi
                
                G.add_node(client, type='Client')
                G.add_node(ap, type='AP')
                G.add_edge(ap, client, rssi=rssi, weight=weight)
                
            # Save Pickle (including band in filename)
            with open(os.path.join(PICKLE_DIR, f"graph_t_{t}_b_{band}.pkl"), 'wb') as f:
                pickle.dump(G, f)
                
            # Save GraphML (including band in filename)
            nx.write_graphml(G, os.path.join(GRAPHML_DIR, f"graph_t_{t}_b_{band}.graphml"))

def main():
    ensure_dirs()
    df = load_data(INPUT_FILE)
    
    # 1. Filter frequent clients
    df = filter_frequent_clients(df)
    
    # 2. Select antenna with highest RSSI
    df = aggregate_highest_rssi(df)
    
    # 3. Interpolate
    interp_data = interpolate_sequences(df)
    
    # 4. Schedule
    df_final = schedule_timesteps(interp_data)
    
    # 5. Save processed data
    print(f"Saving processed data to {OUTPUT_FILE}...")
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    # 6. Generate graphs
    generate_graphs(df_final)
    
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
