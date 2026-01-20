import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_all_metrics(csv_file='training_metrics.csv', output_dir='plots'):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    
    # Set style
    sns.set_theme(style="darkgrid")
    
    # 1. Total Reward Rate per Step
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='total_reward_rate', label='Step Reward')
    # Rolling average
    df['reward_rolling'] = df['total_reward_rate'].rolling(window=10).mean()
    sns.lineplot(data=df, x='step', y='reward_rolling', label='Rolling Avg (10)', color='orange')
    plt.title('Network Rate (Reward) over Time')
    plt.xlabel('Step')
    plt.ylabel('Rate (Mbps)')
    plt.savefig(os.path.join(output_dir, 'reward_rate.png'))
    plt.close()
    
    # 2. Loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    # 3. Learning Rate
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='learning_rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
    plt.close()
    
    # 4. Lagrangian Dual (mu_k) - Per AP Stats
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='mean_mu_k', label='Mean mu_k')
    sns.lineplot(data=df, x='step', y='max_mu_k', label='Max mu_k')
    plt.title('Lagrangian Dual Variable (mu_k) - Per AP')
    plt.ylabel('Penalty Value')
    plt.savefig(os.path.join(output_dir, 'mu_k.png'))
    plt.close()
    
    # 5. Power Usage vs Limit (Per AP)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='mean_power_usage', label='Mean Power Index')
    sns.lineplot(data=df, x='step', y='max_power_usage', label='Max Power Index')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Per-AP Limit (1.0)')
    plt.title('Power Usage (Index) vs Per-AP Limit')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'power_usage.png'))
    plt.close()
    
    # 5. Action Distribution (Stacked Area or similar? Or just counts over time)
    # Counts are per step "sum" across APs? Wait, actions are taken for ALL APs.
    # The metrics log: action_ch_0_count corresponds to how many APs chose Ch 0 in that step.
    
    # Channel Actions - COMMENTED OUT (Data missing in current CSV)
    # plt.figure(figsize=(10, 6))
    # plt.stackplot(df['step'], 
    #               df['action_ch_0_count'], 
    #               df['action_ch_1_count'], 
    #               df['action_ch_2_count'], 
    #               labels=['Ch 1', 'Ch 6', 'Ch 11'], alpha=0.8)
    # plt.title('AP Channel Selection Distribution')
    # plt.xlabel('Step')
    # plt.ylabel('Count of APs')
    # plt.legend(loc='upper left')
    # plt.savefig(os.path.join(output_dir, 'action_distribution_channel.png'))
    # plt.close()
    
    # Power Actions
    # Assuming indices 0, 1, 2 correspond to Low, Med, High
    # Metrics logged: action_pwr_0_count, action_pwr_2_count. Assuming 1 is implied or missing?
    # Let's check columns if 1 is there. If not, we infer.
    # Actually train_gnn.py logged specific keys. Let's plotting what we have.
    
    # Dynamically find power count cols
    # pwr_cols = [c for c in df.columns if 'action_pwr' in c]
    # pwr_cols.sort()
    
    # plt.figure(figsize=(10, 6))
    # plt.stackplot(df['step'], 
    #               *[df[c] for c in pwr_cols], 
    #               labels=[c.replace('action_', '').replace('_count', '') for c in pwr_cols], alpha=0.8)
    # plt.title('AP Power Selection Distribution')
    # plt.xlabel('Step')
    # plt.ylabel('Count of APs')
    # plt.legend(loc='upper left')
    # plt.savefig(os.path.join(output_dir, 'action_distribution_power.png'))
    # plt.close()
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    plot_all_metrics()
