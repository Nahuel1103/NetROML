import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration
CONFIG = {
    'base_dir': 'buildings/1361',
    'analysis_dir': 'buildings/1361/analysis',
    'plots_dir': 'buildings/1361/analysis/plots',
    'target_client': '132.56.56.164.101.173',
    'target_months': ['03', '05', '07', '09', '11'],
    'significance_level': 0.05
}

def ensure_directories():
    """Ensure output directories exist."""
    os.makedirs(CONFIG['analysis_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)

def load_client_data(client_mac, months, base_path):
    """
    Modular data loader.
    Iterates through monthly CSVs and filters for a specific client.
    """
    monthly_data = {}
    for m in months:
        file_path = os.path.join(base_path, f'rssi_2018_{m}.csv')
        if not os.path.exists(file_path):
            print(f"Warning: Data for month {m} not found at {file_path}")
            continue
        
        print(f"Loading data for month {m}...")
        # Efficient loading: only read required columns
        df = pd.read_csv(file_path, usecols=['mac_cliente', 'rssi'])
        client_filtered = df[df['mac_cliente'] == client_mac]['rssi'].values
        
        if len(client_filtered) > 0:
            monthly_data[m] = client_filtered
        else:
            print(f"Info: No records for client {client_mac} in month {m}")
            
    return monthly_data

def perform_anova(monthly_data):
    """
    Performs one-way ANOVA and returns stats + descriptive data.
    """
    if len(monthly_data) < 2:
        return None, "Not enough months with data for ANOVA comparison."

    groups = [data for m, data in monthly_data.items()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    descriptive_stats = []
    for m, data in monthly_data.items():
        descriptive_stats.append({
            'Month': m,
            'Count': len(data),
            'Mean': np.mean(data),
            'StdDev': np.std(data, ddof=1),
            'Variance': np.var(data, ddof=1)
        })
    
    return {'f_stat': f_stat, 'p_val': p_val, 'descriptive': pd.DataFrame(descriptive_stats)}, None

def visualize_results(monthly_data, client_mac):
    """
    Generates a boxplot comparison using Seaborn.
    """
    # Prepare data for long-form DataFrame (required for Seaborn)
    plot_rows = []
    for m, rssis in monthly_data.items():
        for r in rssis:
            plot_rows.append({'Month': m, 'RSSI': r})
    
    df_plot = pd.DataFrame(plot_rows)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.boxplot(x='Month', y='RSSI', data=df_plot, palette="Set2")
    plt.title(f'RSSI Distribution Across Months - Client {client_mac}')
    plt.ylabel('RSSI (dBm)')
    plt.xlabel('Month (2018)')
    
    output_path = os.path.join(CONFIG['plots_dir'], 'anova_boxplot.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_report(client_mac, results, plot_path):
    """
    Generates a modular Markdown report with findings.
    """
    f_stat = results['f_stat']
    p_val = results['p_val']
    stats_df = results['descriptive']
    
    is_significant = p_val < CONFIG['significance_level']
    conclusion = (
        "There IS a statistically significant difference in mean RSSI across the selected months."
        if is_significant else
        "There is NO statistically significant difference in mean RSSI across the selected months."
    )

    report = f"""# ANOVA Statistical Analysis Report
## Target: User {client_mac}
- **Objective**: Compare the mean RSSI values across multiple months to detect significant signal variations.
- **Months Analysed**: {', '.join(CONFIG['target_months'])}

## Descriptive Statistics
{stats_df.to_markdown(index=False)}

## ANOVA Test Results (One-Way)
- **F-statistic**: {f_stat:.4f}
- **p-value**: {p_val:.4e}
- **Significance Level ($\alpha$)**: {CONFIG['significance_level']}

### Conclusion
**{conclusion}**
(Result based on p-value {'<' if is_significant else '>='} {CONFIG['significance_level']}).

## Visualization
The boxplot below visually confirms the differences (or lack thereof) in signal distribution patterns between the months.
(Plot saved at: {plot_path})
"""
    report_path = os.path.join(CONFIG['analysis_dir'], 'anova_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    return report_path

def main():
    ensure_directories()
    
    # 1. Load Data
    data = load_client_data(CONFIG['target_client'], CONFIG['target_months'], CONFIG['base_dir'])
    
    # 2. Perform Stats
    results, error = perform_anova(data)
    if error:
        print(f"Error: {error}")
        return

    # 3. Visualize
    plot_p = visualize_results(data, CONFIG['target_client'])
    
    # 4. Report
    report_p = generate_report(CONFIG['target_client'], results, plot_p)
    
    print("\n" + "="*40)
    print(" ANOVA ANALYSIS COMPLETE")
    print("="*40)
    print(f"F-stat:  {results['f_stat']:.4f}")
    print(f"p-value: {results['p_val']:.4e}")
    print(f"Report:  {report_p}")
    print(f"Plot:    {plot_p}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
