import pandas as pd
import numpy as np
import os
import random
from scipy.interpolate import interp1d

# Constants
INPUT_FILE = 'buildings/990/rssi_2018_03.csv'
OUTPUT_FILE = 'buildings/990/processed_rssi_2018_03.csv'
TARGET_LEN = 20
GAP = 5
MAX_START_OFFSET = 100 # Maximum random start time offset

def load_data(filepath):
    """Loads the CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def process_connection_sequence(group, target_len=TARGET_LEN):
    """
    Processes a single Client-AP connection group.
    1. Aggregates multi-antenna clumps into single points (max RSSI).
    2. Interpolates if sequence is shorter than target_len.
    """
    # 1. Deduplicate/Aggregate simultaneous readings (Antenna diversity)
    # Strategy: Consecutive identical timestamps generally imply simultaneous readings.
    # However, the input file has NO timestamps.
    # We assume the file order is sequential.
    # We need to collapse 'bursts' of records into single time steps.
    # The user mentioned ~4 records per connection often (ant0, ant1... etc).
    # Since we lack real time, we'll take a simple approach:
    # Just take the raw list of RSSI. If the user implies these 4 records ARE the same instant,
    # we should max-pool them.
    # Heuristic: We will iterate and max-pool adjacent records if they are 'close' in index
    # BUT without time, 'close' is hard.
    # Let's assume the user meant the raw rows are just a stream.
    # ACTUALLY, checking the file sample:
    # 2: clientA, ap1, 0, 0, -97
    # 3: clientA, ap1, 0, 1, -99
    # 4: clientA, ap1, 0, 0, -97
    # 5: clientA, ap1, 0, 1, -99
    # This looks like repeated samples or multi-antenna.
    # Let's group by "blocks".
    # Since we want to construct a timeline, let's treat every unique combinations of (banda, antena)
    # as parallel sensors.
    # Simplification: Just allow the stream to be the timeline, but if it's too short, interpolate.
    
    # Wait, the user said: "Agrupar las mediciones por cliente–AP para generar una conexión por mes"
    # And "Construir los datos de RSSI de cada conexión de forma independiente."
    
    raw_rssi = group['rssi'].values
    
    # If we have very few points, we might just use them all.
    # If they are duplicates (same value, diff antenna), we might want to average/max them.
    # Let's perform a 'compaction' -> neighboring values that are part of the same scan.
    # How to detect a scan? Maybe sets of 2 or 4?
    # Let's assume grouping by index/row_number blocks isn't reliable without time.
    # Let's just process the raw signal.
    # Optionally: Smooth/Max-pool every N samples?
    # Given the small sample:
    # 8.212... has 16 rows.
    # It has distinct blocks of readings.
    # Let's assume the RAW sequence is the valid timeline, but we want to STRETCH it to at least TARGET_LEN.
    
    # Let's try to detect if we should compact.
    # Rows 2-5 are for same AP-Client.
    # 0,0,-97; 0,1,-99; 0,0,-97; 0,1,-99. This looks like 2 samples, each with 2 antennas.
    # Real effective samples = 2.
    # Let's try to compact by taking MAX of every chunk of 2? Or just raw?
    # User said: "Eliminar completamente la referencia temporal".
    # Let's stick to: Interpolate 'raw_rssi' to 'target_len'.
    
    # However, to avoid high frequency noise from antenna switching (e.g. -97, -99, -97, -99), 
    # taking the max of pairs might be better if we assume 2 antennas.
    # Let's assume dual antenna (0,1) is standard here.
    # Let's grouping by (row_number // 2)? No, dangerous if missing rows.
    
    # Let's just treat the sequence of RSSI values as the signal.
    y = raw_rssi
    x = np.linspace(0, 1, num=len(y))
    
    # Interpolation
    if len(y) < target_len:
        f = interp1d(x, y, kind='linear')
        new_x = np.linspace(0, 1, num=target_len)
        new_y = f(new_x)
        return new_y.astype(int) # RSSI is usually int
    else:
        # If longer, we can keep it or downsample?
        # User prompt implies "Interpolar... cuando haya POCAS mediciones".
        # So we keep long ones as is.
        return y

def main():
    print(f"Loading {INPUT_FILE}...")
    df = load_data(INPUT_FILE)
    
    # Group by Client-AP
    # Note: We must preserve the order of appearance or simple group?
    # Default pandas groupby preserves order of keys? sort=False
    grouped = df.groupby(['mac_cliente', 'mac_ap'], sort=False)
    
    processed_rows = []
    
    # Dictionary to track the next available time slot for each client
    client_clocks = {} # { client_mac: int_time_step }
    
    print(f"Processing {len(grouped)} connection groups...")
    
    # We iterate groups.
    # IMPORTANT: The user said "Agrupar... para generar UNA conexión por mes".
    # This implies we merge ALL data for (Client, AP) into one sequence.
    
    for (client, ap), group in grouped:
        rssi_seq = process_connection_sequence(group)
        
        # Scheduling
        if client not in client_clocks:
             # Randomize start time for the first connection of each client
             # to distribute them across time steps.
             client_clocks[client] = random.randint(0, MAX_START_OFFSET)
             
        start_time = client_clocks[client]
        duration = len(rssi_seq)
        
        # Create rows
        for t_offset, rssi_val in enumerate(rssi_seq):
            processed_rows.append({
                'mac_cliente': client,
                'mac_ap': ap,
                'rssi': rssi_val,
                'time_step': start_time + t_offset
            })
            
        # Update clock with GAP
        client_clocks[client] = start_time + duration + GAP
        
    # Convert to DataFrame
    result_df = pd.DataFrame(processed_rows)
    
    # Verify strict non-overlap per client (inherent by construction, but good to check)
    # And allow overlap across clients (inherent by independent clocks)
    
    print(f"Saving to {OUTPUT_FILE}...")
    result_df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")
    print("Sample output:")
    print(result_df.head(10))

if __name__ == "__main__":
    main()
