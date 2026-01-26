import torch
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from network_graph_env import NetworkGraphEnv
from gnn_model import GNN
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "buildings"

def mu_update(mu_k, power_constr, eps):
    mu_k = mu_k.detach()
    # mean over dim 0 (batch dimension)
    # If power_constr is [1, Num_APs], this returns [Num_APs]
    mu_k_update = eps * torch.mean(power_constr, dim = 0) 
    mu_k = mu_k + mu_k_update
    mu_k = torch.max(mu_k, torch.tensor(0.0))
    return mu_k

def train():
    # 1. Initialize API
    env = NetworkGraphEnv(data_root=DATA_ROOT, building_id=990)
    num_node_features = 4 
    hidden_channels = 32
    num_aps = env.num_aps
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.backends.metal.is_available():
        device = torch.device('metal')
    else:
        device = torch.device('cpu')
    
    model = GNN(num_node_features, hidden_channels, num_aps, K=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Constrained Optimization Params (Lagrangian) - Per AP
    # ---------------------------------------------------------
    # Limit: Each AP should not exceed this average power in dBm
    P_LIMIT_PER_AP = 15.0  # dBm - between Low(10) and Medium(17)
    
    # Dual Variable mu_k: Vector of size [num_aps]
    mu_k = torch.zeros(num_aps, device=device) 
    epsilon = 0.1 # Step size for dual variable update
    
    # Power levels for constraint calculation
    tx_powers_dbm = torch.tensor([10.0, 17.0, 23.0], device=device)
    
    print(f"Starting REINFORCE with Per-AP Constraints on device: {device}")
    print(f"Per-AP Power Limit: {P_LIMIT_PER_AP} dBm")
    
    metrics = []
    num_episodes = 100
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_rate = 0
        step_count = 0
        
        while not done:
            data = obs.to(device)
            
            # 1. Forward Pass
            ch_logits, pwr_logits = model(data.x, data.edge_index, data.edge_attr)
            
            # 2. Action Sampling
            ch_dist = Categorical(logits=ch_logits)
            pwr_dist = Categorical(logits=pwr_logits)
            
            ch_actions = ch_dist.sample() # [Num_APs]
            pwr_actions = pwr_dist.sample() # [Num_APs]
            
            log_prob_ch = ch_dist.log_prob(ch_actions)
            log_prob_pwr = pwr_dist.log_prob(pwr_actions)
            
            action = torch.stack((ch_actions, pwr_actions), dim=1).flatten().cpu().numpy()
            
            # 3. Step Environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # ---------------------------------------------------------
            # Constraint Calculation (Per AP) - Using Physical Power
            # ---------------------------------------------------------
            # Convert action indices to actual power in dBm
            current_power_dbm = tx_powers_dbm[pwr_actions]  # [Num_APs]
            
            # Vector Constraint: power_dbm[i] - P_LIMIT <= 0
            constraint_val = current_power_dbm - P_LIMIT_PER_AP  # [Num_APs]
            
            # Update dual variables (add batch dimension for mu_update function)
            mu_k = mu_update(mu_k, constraint_val.unsqueeze(0), epsilon)
            
            # 4. Calculate Loss (Primal-Dual)
            scaled_reward = reward / 1000.0
            
            # Cost = -Reward + Sum(mu_k_i * constraint_val_i)
            constraint_penalty = torch.sum(mu_k * constraint_val)
            cost = -scaled_reward + constraint_penalty
            
            # REINFORCE Loss
            loss = cost.detach() * (torch.sum(log_prob_ch) + torch.sum(log_prob_pwr))
            
            # 5. Update Primal
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_rate += reward
            step_count += 1
            obs = next_obs
            
            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            metrics.append({
                "episode": episode + 1,
                "step": step_count,
                "total_reward_rate": reward,
                "loss": loss.item(),
                "learning_rate": current_lr,
                "mean_mu_k": mu_k.mean().item(),
                "max_mu_k": mu_k.max().item(),
                "active_constraints": (mu_k > 0).sum().item(),
                "mean_power_dbm": current_power_dbm.mean().item(),
                "max_power_dbm": current_power_dbm.max().item(),
                "action_pwr_0_count": (pwr_actions == 0).sum().item(),
                "action_pwr_1_count": (pwr_actions == 1).sum().item(),
                "action_pwr_2_count": (pwr_actions == 2).sum().item()
            })
            
        print(f"Episode {episode+1}: Rate = {total_rate:.2f} Mbps | Mean mu_k = {mu_k.mean():.4f} | Max mu_k = {mu_k.max():.4f} | Active Constraints = {(mu_k > 0).sum().item()} | Mean Power = {mu_k.mean().item():.1f} dBm")
        
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("training_metrics.csv", index=False)
    print("Training metrics saved.")

if __name__ == "__main__":
    train()
