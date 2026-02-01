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

def compute_returns(rewards, gamma=0.99):
    """Calcula retornos descontados."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

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
    seed = 314
    env = NetworkGraphEnv(data_root=data_root, building_id=990, random_seed=seed)
    
    hidden_channels = 64 # Increased for Transformer/GAT
    num_aps = env.num_aps
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Training on device: {device}")

    num_episodes = 200
    
    model = GNN(hidden_channels, num_aps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_episodes)
    
    # Baseline for REINFORCE
    running_reward_mean = 0.0
    alpha = 0.1
    
    best_reward = -float('inf')
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    
    metrics = []
    
    
    start_time_total = time.time()
    
    for episode in range(num_episodes):
        obs, _ = env.reset() # Obs is HeteroData
        done = False
        total_rate = 0.0
        step_count = 0
        
        log_probs_ch = []
        log_probs_pwr = []
        all_ch_actions = []
        all_pwr_actions = []
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
            
            all_ch_actions.append(ch_actions)
            all_pwr_actions.append(pwr_actions)
            
            # Form action
            action_tensor = torch.stack((ch_actions, pwr_actions), dim=1).flatten()
            action = action_tensor.cpu().numpy()
            
            # 3. Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            fairness_scores.append(info.get('fairness_index', 0))
            
            total_rate += info.get('total_rate', 0) # Track real throughput, not log reward
            step_count += 1
            obs = next_obs
        
        # --- Update ---

        # 1. Compute returns
        returns = compute_returns(rewards, gamma=0.99)
        G = returns[0].item()

        # 2. Update baseline
        if episode == 0:
            running_reward_mean = G
        else:
            running_reward_mean = (1 - alpha) * running_reward_mean + alpha * G

        # 3. Compute advantages (normalized)
        advantages = returns - running_reward_mean
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. Policy loss por step
        policy_losses = []
        for t in range(len(rewards)):
            lp_ch = log_probs_ch[t].sum()
            lp_pwr = log_probs_pwr[t].sum()
            policy_losses.append(-advantages[t] * (lp_ch + lp_pwr))

        policy_loss = torch.stack(policy_losses).mean()

        # 5. Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()

        # 6. Gradient clipping (OPCIÓN B)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        
        # --- Model Saving ---
        if G > best_reward:
            best_reward = G
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            
        if (episode + 1) % 50 == 0:
             torch.save(model.state_dict(), save_dir / f"model_ep_{episode+1}.pt")
        
        # --- Action Analysis ---
        ep_ch_actions = torch.cat(all_ch_actions).cpu().numpy()
        ep_pwr_actions = torch.cat(all_pwr_actions).cpu().numpy()
        
        # --- Logging ---
        ep_duration = time.time() - ep_start_time
        metrics.append({
            "episode": episode + 1,
            "total_rate_mbps": total_rate,
            "mean_fairness": np.mean(fairness_scores),
            "loss": policy_loss.item(),
            "reward_sum": G,
            "grad_norm": grad_norm.item(),
            "lr": optimizer.param_groups[0]['lr'],
            "duration_sec": ep_duration,
            "mean_pwr_idx": ep_pwr_actions.mean(),
        })
        
        if (episode+1) % 10 == 0:
            print(f"Ep {episode+1}: Rew={G:.1f} | Rate={total_rate:.1f} | Fair={np.mean(fairness_scores):.2f} | "
                  f"Loss={policy_loss.item():.2f} | Grad={grad_norm:.2f} | Time={ep_duration:.2f}s")

        # --- Snapshot Saving ---
        if (episode + 1) % 20 == 0:
            snapshot = {
                "episode": episode + 1,
                "obs": obs, 
                "info": info,
                "all_ch_actions": ep_ch_actions,
                "all_pwr_actions": ep_pwr_actions
            }
            torch.save(snapshot, save_dir / f"snapshot_ep_{episode+1}.pt")

    print(f"Training finished in {time.time() - start_time_total:.1f}s")
    
    # Save Final
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    
    df_metrics = pd.DataFrame(metrics)
    output_path = "training_metrics.csv"
    df_metrics.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    train()

