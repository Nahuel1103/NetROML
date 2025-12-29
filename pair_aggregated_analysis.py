import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Target Configuration
BASE_DIR = 'buildings/1361'
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis_aggregated')
PLOTS_DIR = os.path.join(ANALYSIS_DIR, 'plots')
MONTHS = ['03', '05', '07', '09', '11']
CLIENT = '132.56.56.164.101.173'
AP = '136.29.252.141.130.32'

def load_aggregated_data():
    all_rssi = []
    for m in MONTHS:
        fname = f'rssi_2018_{m}.csv'
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found.")
            continue
        
        print(f"Filtering {fname} for pair...")
        # Load only necessary columns to be efficient
        df = pd.read_csv(fpath, usecols=['mac_cliente', 'mac_ap', 'rssi'])
        df_filtered = df[(df['mac_cliente'] == CLIENT) & (df['mac_ap'] == AP)]
        all_rssi.extend(df_filtered['rssi'].tolist())
    
    return np.array(all_rssi)

def perform_aggregated_stats(rssi_values):
    if len(rssi_values) == 0:
        print("No data found for this pair in the target months.")
        return

    # Descriptive Stats
    mean = np.mean(rssi_values)
    variance = np.var(rssi_values, ddof=1) # Sample variance
    std_dev = np.std(rssi_values, ddof=1)
    count = len(rssi_values)
    median = np.median(rssi_values)
    
    # Confidence Interval (95%)
    ci_low, ci_high = stats.t.interval(0.95, df=count-1, loc=mean, scale=stats.sem(rssi_values))

    # --- NEW: Statistical Tests ---
    
    # 1. One-sample T-test (Null Hypothesis: Mean = -75 dBm)
    pop_mean_ref = -75.0
    t_stat, t_p_val = stats.ttest_1samp(rssi_values, pop_mean_ref)

    # 2. Chi-squared Normality Test (D'Agostino's K^2)
    # This test combines skew and kurtosis to produce an omnibus test of normality.
    chi_stat, chi_p_val = stats.normaltest(rssi_values)

    print("\n" + "="*40)
    print(" AGGREGATED STATISTICAL RESULTS")
    print("="*40)
    print(f"Pair: Client({CLIENT}) | AP({AP})")
    print(f"Months: {', '.join(MONTHS)}")
    print("-" * 40)
    print(f"Total Measurements (N): {count}")
    print(f"Mean RSSI:             {mean:.4f} dBm")
    print(f"Median RSSI:           {median:.4f} dBm")
    print(f"Sample Variance:       {variance:.4f}")
    print(f"Std Deviation:         {std_dev:.4f}")
    print(f"95% CI for Mean:       ({ci_low:.4f}, {ci_high:.4f})")
    print("-" * 40)
    print(" HYPOTHESIS TESTING")
    print("-" * 40)
    print(f"One-sample T-test (vs {pop_mean_ref} dBm):")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {t_p_val:.4e}")
    print(f"Chi-squared Normality Test (K^2):")
    print(f"  chi2-stat:   {chi_stat:.4f}, p-value: {chi_p_val:.4e}")
    print("="*40 + "\n")

    # Generate Report
    report_content = f"""# Aggregated RSSI Analysis Report
## Pair Identification
- **Client**: {CLIENT}
- **AP**: {AP}
- **Months Included**: {', '.join(MONTHS)}

## Statistics (Aggregated)
- **Sample Size (N)**: {count}
- **Mean RSSI**: {mean:.4f} dBm
- **Median RSSI**: {median:.4f} dBm
- **Variance**: {variance:.4f}
- **Standard Deviation**: {std_dev:.4f}
- **95% Confidence Interval**: [{ci_low:.4f}, {ci_high:.4f}]

## Hypothesis Testing
### Student's T-test (One-sample)
- **Null Hypothesis ($H_0$)**: The population mean is {pop_mean_ref} dBm.
- **Statistic**: t = {t_stat:.4f}
- **p-value**: {t_p_val:.4e}
- **Result**: {'Reject $H_0$' if t_p_val < 0.05 else 'Fail to reject $H_0$'} (at 5% significance level).

### Chi-squared Normality Test
- **Null Hypothesis ($H_0$)**: The distribution is Normal.
- **Statistic**: $\chi^2$ = {chi_stat:.4f}
- **p-value**: {chi_p_val:.4e}
- **Result**: {'Reject $H_0$ (Not Normal)' if chi_p_val < 0.05 else 'Fail to reject $H_0$ (Likely Normal)'}.

## Findings
The distribution of RSSI values shows a mean of {mean:.4f} dBm. The T-test indicates that this {'is' if t_p_val < 0.05 else 'is not'} significantly different from {pop_mean_ref} dBm. Additionally, the Chi-squared normality test suggests the data {'is not' if chi_p_val < 0.05 else 'is potentially'} normally distributed.
"""
    with open(os.path.join(ANALYSIS_DIR, 'aggregated_report.md'), 'w') as f:
        f.write(report_content)

def visualize_distribution(rssi_values):
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Distribution Plot
    sns.histplot(rssi_values, kde=True, bins=20, color='royalblue', alpha=0.6)
    
    # Overlay mean line
    plt.axvline(np.mean(rssi_values), color='red', linestyle='--', label=f'Mean: {np.mean(rssi_values):.2f}')
    
    plt.title(f'Aggregated RSSI Distribution - Top Pair (Months: {", ".join(MONTHS)})')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'aggregated_rssi_distribution.png'))
    plt.close()
    print(f"Visualization saved to {os.path.join(PLOTS_DIR, 'aggregated_rssi_distribution.png')}")

def main():
    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    rssi_data = load_aggregated_data()
    perform_aggregated_stats(rssi_data)
    visualize_distribution(rssi_data)

if __name__ == "__main__":
    main()
