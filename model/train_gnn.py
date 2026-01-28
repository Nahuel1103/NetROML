import torch
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
import torch.optim as optim

from gnn_model import GNN
from network_graph_env import NetworkGraphEnv

def train():
    # 0. Path Setup
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    data_root = project_root
    
    # 1. Initialize Env
    # Check if 'buildings' folder exists in project root, otherwise use root (some users put 990 in root)
    if (project_root / "buildings").exists():
        data_root = project_root / "buildings"
    else:
        data_root = project_root

    print(f"Loading environment with data root: {data_root}")
    env = NetworkGraphEnv(data_root=data_root, building_id=990)
    
    hidden_channels = 64 # Increased for Transformer/GAT
    num_aps = env.num_aps
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Training on device: {device}")
    
    model = GNN(hidden_channels, num_aps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Baseline for REINFORCE
    running_reward_mean = 0.0
    alpha = 0.1
    
    metrics = []
    num_episodes = 200
    
    start_time_total = time.time()
    
    for episode in range(num_episodes):
        obs, _ = env.reset() # Obs is HeteroData
        done = False
        total_rate = 0.0
        step_count = 0
        
        log_probs_ch = []
        log_probs_pwr = []
        rewards = []
        fairness_scores = []
        
        ep_start_time = time.time()
        
        while not done:
            data = obs.to(device)
            
            # 1. Forward Pass
            # Pass x_dict, edge_index_dict, edge_attr_dict
            ch_logits, pwr_logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            
            # 2. Action Sampling
            ch_dist = Categorical(logits=ch_logits)
            pwr_dist = Categorical(logits=pwr_logits)
            
            ch_actions = ch_dist.sample()
            pwr_actions = pwr_dist.sample()
            
            log_probs_ch.append(ch_dist.log_prob(ch_actions))
            log_probs_pwr.append(pwr_dist.log_prob(pwr_actions))
            
            # Form action
            action_tensor = torch.stack((ch_actions, pwr_actions), dim=1).flatten()
            action_np = action_tensor.cpu().numpy()
            
            # 3. Step
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            rewards.append(reward)
            fairness_scores.append(info.get('fairness_index', 0))
            
            total_rate += info.get('total_rate', 0) # Track real throughput, not log reward
            step_count += 1
            obs = next_obs
        
        # --- Update ---
        G = sum(rewards)
        
        if episode == 0:
            running_reward_mean = G
        else:
            running_reward_mean = (1 - alpha) * running_reward_mean + alpha * G
            
        advantage = G - running_reward_mean
        
        optimizer.zero_grad()
        
        # Sum log probs
        sum_log_probs = torch.cat(log_probs_ch).sum() + torch.cat(log_probs_pwr).sum()
        
        # Loss = - Advantage * SumLogProbs
        loss = -advantage * sum_log_probs
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        ep_duration = time.time() - ep_start_time
        metrics.append({
            "episode": episode + 1,
            "total_rate_mbps": total_rate,
            "mean_fairness": np.mean(fairness_scores),
            "loss": loss.item(),
            "reward_sum": G,
            "duration_sec": ep_duration
        })
        
        if (episode+1) % 10 == 0:
            print(f"Ep {episode+1}: Rew={G:.1f} | Rate={total_rate:.1f} | Fair={np.mean(fairness_scores):.2f} | Time={ep_duration:.2f}s")

    print(f"Training finished in {time.time() - start_time_total:.1f}s")
    df_metrics = pd.DataFrame(metrics)
    output_path = "training_metrics.csv"
    df_metrics.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    train()

