import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ruptures as rpt

# Configuration
CONFIG = {
    'base_dir': 'buildings/1361',
    'analysis_dir': 'buildings/1361/analysis_ruptures',
    'plots_dir': 'buildings/1361/analysis_ruptures/plots',
    'client_mac': '32.45.7.240.187.6',  # Using the top pair client from previous analysis
    'ap_mac': '136.29.252.170.206.224', # Using the top pair AP from previous analysis
    'months': ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11'],
    # Ruptures settings
    'model': 'l2',  # "l2", "rbf", "linear", "normal", "ar"
    'penalty': 10   # Penalty value for Pelt search method
}

def ensure_directories():
    os.makedirs(CONFIG['analysis_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)

def load_data():
    """
    Loads RSSI data for the specific Client-AP pair across months.
    Concatenates them chronologically.
    """
    all_rssi = []
    print(f"Loading data for Pair: Client({CONFIG['client_mac']}) - AP({CONFIG['ap_mac']})")
    
    for m in CONFIG['months']:
        fpath = os.path.join(CONFIG['base_dir'], f'rssi_2018_{m}.csv')
        if not os.path.exists(fpath):
            print(f"  - Month {m}: File not found.")
            continue
            
        try:
            # Optimize load: only needed columns
            df = pd.read_csv(fpath, usecols=['mac_cliente', 'mac_ap', 'rssi'])
            
            # Filter for the pair
            pair_data = df[
                (df['mac_cliente'] == CONFIG['client_mac']) & 
                (df['mac_ap'] == CONFIG['ap_mac'])
            ]['rssi'].tolist()
            
            if pair_data:
                print(f"  - Month {m}: Found {len(pair_data)} samples.")
                all_rssi.extend(pair_data)
            else:
                print(f"  - Month {m}: No samples for this pair.")
                
        except Exception as e:
            print(f"  - Month {m}: Error loading ({e})")

    return np.array(all_rssi)

def perform_change_point_detection(signal):
    """
    Apply Ruptures Pelt search method to find change points in the mean.
    """
    if len(signal) < 10:
        print("Not enough data points for Change Point Detection.")
        return None

    print(f"\nRunning Change Point Detection (Model: {CONFIG['model']}, Penalty: {CONFIG['penalty']})...")
    
    # Change point detection
    algo = rpt.Pelt(model=CONFIG['model']).fit(signal)
    result = algo.predict(pen=CONFIG['penalty'])
    
    # result includes the end index of the signal, so we have N change points + end
    # Let's count actual changes
    n_changes = len(result) - 1
    print(f"Detected {n_changes} change points at indices: {result[:-1]}")
    
    return result

def visualize_change_points(signal, bkps):
    """
    Visualize the signal and the detected change points.
    """
    plt.figure(figsize=(15, 6))
    
    # Plotting signal
    rpt.display(signal, bkps, figsize=(15, 6))
    plt.title(f'RSSI Change Point Detection\nClient: {CONFIG["client_mac"]} | AP: {CONFIG["ap_mac"]}')
    plt.xlabel('Sample Index (Time ordered by Month)')
    plt.ylabel('RSSI (dBm)')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    out_path = os.path.join(CONFIG['plots_dir'], 'change_points.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def generate_report(signal, bkps):
    """
    Generate a markdown report summarizing the findings.
    """
    n_changes = len(bkps) - 1
    
    # Calculate stats for each segment
    segments_summary = []
    start_idx = 0
    for i, end_idx in enumerate(bkps):
        segment_data = signal[start_idx:end_idx]
        if len(segment_data) > 0:
            seg_mean = np.mean(segment_data)
            seg_std = np.std(segment_data)
            segments_summary.append(f"| {i+1} | {start_idx} to {end_idx-1} | {len(segment_data)} | {seg_mean:.2f} | {seg_std:.2f} |")
        start_idx = end_idx

    report = f"""# Change Point Analysis Report

## Trace Info
- **Client**: `{CONFIG['client_mac']}`
- **AP**: `{CONFIG['ap_mac']}`
- **Total Samples**: {len(signal)}
- **Months**: {', '.join(CONFIG['months'])}

## Detection Results
- **Algorithm**: Pelt
- **Model**: {CONFIG['model']} (Least Squared Error)
- **Penalty**: {CONFIG['penalty']}
- **Detected Change Points**: {n_changes}

### Segment Statistics
| Segment | Index Range | Samples | Mean RSSI (dBm) | Std Dev |
| :--- | :--- | :--- | :--- | :--- |
{chr(10).join(segments_summary)}

## Conclusion
The analysis detected **{n_changes}** points where the statistical properties (mean) of the RSSI signal changed significantly.
{
"The signal mean appears relatively stable." if n_changes == 0 else 
"The signal shows distinct shifts in mean RSSI, suggesting environmental changes or mobility behavior."
}

## Visualization
![Change Points](plots/change_points.png)
"""
    
    report_path = os.path.join(CONFIG['analysis_dir'], 'ruptures_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    return report_path

import random

def select_random_pair(min_samples=50):
    """
    Selects a random Client-AP pair that has at least `min_samples` in a random month.
    """
    print("Selecting a random valid pair...")
    shuffled_months = CONFIG['months'].copy()
    random.shuffle(shuffled_months)
    
    for m in shuffled_months:
        fpath = os.path.join(CONFIG['base_dir'], f'rssi_2018_{m}.csv')
        if not os.path.exists(fpath):
            continue
            
        print(f"  - Sampling month {m}...")
        try:
            df = pd.read_csv(fpath, usecols=['mac_cliente', 'mac_ap'])
            # Group by pair and count
            pair_counts = df.groupby(['mac_cliente', 'mac_ap']).size()
            
            # Filter candidates
            candidates = pair_counts[pair_counts >= min_samples].index.tolist()
            
            if candidates:
                selected = random.choice(candidates)
                print(f"  - Selected Pair: Client={selected[0]}, AP={selected[1]} (Samples in {m}: {pair_counts[selected]})")
                return selected[0], selected[1]
                
        except Exception as e:
            print(f"    Error reading month {m}: {e}")

    print("  - Could not find a pair with sufficient data.")
    return None, None

def main():
    ensure_directories()
    
    # 1. Select Random Pair
    client, ap = select_random_pair(min_samples=50)
    if not client:
        return
        
    # Update Config for this run
    CONFIG['client_mac'] = client
    CONFIG['ap_mac'] = ap
    
    # 2. Load Data
    rssi_signal = load_data()
    if len(rssi_signal) == 0:
        print("No data found!")
        return

    # 3. Detect Changes
    bkps = perform_change_point_detection(rssi_signal)
    if bkps is None:
        return

    # 4. Visualize
    plot_path = visualize_change_points(rssi_signal, bkps)
    print(f"Plot saved to: {plot_path}")

    # 5. Generate Report
    report_path = generate_report(rssi_signal, bkps)
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
