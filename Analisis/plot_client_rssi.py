import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_client_rssi(month, client_mac, building_id="990"):
    # Construct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, f"../buildings/{building_id}/rssi_{month}.csv")
    input_file = os.path.abspath(input_file)

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Filter by client
    df_client = df[df['mac_cliente'] == client_mac].copy()

    if df_client.empty:
        print(f"No data found for client {client_mac} in {month}")
        return

    print(f"Found {len(df_client)} records for client {client_mac}")

    # Convert time to datetime for better plotting
    # Format in CSV is date (YYYY-MM-DD), time (HH:MM)
    try:
        df_client['datetime'] = pd.to_datetime(df_client['date'] + ' ' + df_client['time'])
        df_client.sort_values('datetime', inplace=True)
    except Exception as e:
        print(f"Error parsing date/time: {e}")
        return

    # Get unique APs
    mac_aps = df_client['mac_ap'].unique()
    print(f"Client connected to {len(mac_aps)} APs: {mac_aps}")

    # Plot
    for mac_ap in mac_aps:
        subset = df_client[df_client['mac_ap'] == mac_ap]
        
        plt.figure(figsize=(14, 7))
        
        # Plot using scatter
        # Hue for Band, Style for Antenna
        sns.scatterplot(data=subset, x='datetime', y='rssi', hue='banda', style='antena', palette='viridis', s=100)
        
        plt.title(f"RSSI Variation for Client {client_mac}\nAP: {mac_ap} ({month})")
        plt.xlabel("Time")
        plt.ylabel("RSSI (dBm)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_filename = f"rssi_plot_{month}_{client_mac.replace('.', '_')}_{mac_ap.replace('.', '_')}.png"
        output_path = os.path.join(script_dir, output_filename)
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RSSI for a client")
    parser.add_argument("--month", required=True, help="Month format YYYY_MM (e.g. 2018_07)")
    parser.add_argument("--client", required=True, help="Client MAC address")
    parser.add_argument("--building", default="990", help="Building ID")
    
    args = parser.parse_args()
    
    plot_client_rssi(args.month, args.client, args.building)
