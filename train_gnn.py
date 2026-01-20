import torch
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from network_graph_env import NetworkGraphEnv
from gnn_model import GNN
import os

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
    env = NetworkGraphEnv('rssi_2018_08_processed.csv')
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
    
    # ---------------------------------------------------------
    # Constrained Optimization Params (Lagrangian) - Per AP
    # ---------------------------------------------------------
    # Limit: Each AP should not exceed 'Medium' power on average (Index 1.0)
    P_LIMIT_PER_AP = 1.0 
    
    # Dual Variable mu_k: Vector of size [num_aps]
    mu_k = torch.zeros(num_aps, device=device) 
    epsilon = 0.05 # Increased step size for stronger reaction
    
    print(f"Starting REINFORCE with Per-AP Constraints on device: {device}")
    print(f"Per-AP Power Limit Index: {P_LIMIT_PER_AP}")
    
    metrics = []
    num_episodes = 100 # Increased slightly for testing
    
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
            # Constraint Calculation (Per AP)
            # ---------------------------------------------------------
            # pwr_actions is [Num_APs]
            current_power_usage = pwr_actions.float()
            
            # Vector Constraint: usage[i] - P_LIMIT <= 0
            # Ensure it has a batch dim for mu_update if needed, but here it's 1D [Num_APs] 
            # and our mu_k is [Num_APs]. dim=0 in mu_update usually assumes batch. 
            # Since we have no batch dim (it is just nodes), we interpret dim=0 as batch if input was [Batch, Nodes].
            # Here input is [Nodes]. mean(dim=0) would collapse Nodes! 
            # We want Element-wise update.
            # Reference `mu_update` does `mean(dim=0)`.
            # If we pass `constraint_val.unsqueeze(0)` (Shape [1, Num_APs]), 
            # then `mean(dim=0)` returns [Num_APs], which is what we want (averaging over batch of 1).
            
            constraint_val = current_power_usage - P_LIMIT_PER_AP # [Num_APs]
            
            # Use separate function as requested
            mu_k = mu_update(mu_k, constraint_val.unsqueeze(0), epsilon) # Input [1, Num_APs]
            
            # 4. Calculate Loss (Primal-Dual)
            scaled_reward = reward / 1000.0
            
            # Cost = -Reward + Sum(mu_k_i * constraint_val_i)
            # Dot product or sum of element-wise product
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
                "mean_power_usage": current_power_usage.mean().item(),
                "max_power_usage": current_power_usage.max().item(),
                "action_pwr_0_count": (pwr_actions == 0).sum().item(),
                "action_pwr_1_count": (pwr_actions == 1).sum().item(),
                "action_pwr_2_count": (pwr_actions == 2).sum().item()
            })
            
        print(f"Episode {episode+1}: Rate = {total_rate:.2f} Mbps | Mean mu_k = {mu_k.mean():.4f} | Max mu_k = {mu_k.max():.4f}")
        
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("training_metrics.csv", index=False)
    print("Training metrics saved.")

if __name__ == "__main__":
    train()
