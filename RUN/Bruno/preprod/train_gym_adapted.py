from torch_geometric.nn import LayerNorm, Sequential
from torch_geometric.nn.conv import MessagePassing

import random
import pickle
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TAGConv
from torch_geometric.nn import GCNConv
import os
import matplotlib.pyplot as plt

import scipy.io

from plot_results_torch import plot_results
from gnn import GNN
from utils import graphs_to_tensor
from utils import get_gnn_inputs
from utils import graphs_to_tensor_synthetic
from realtime_plotter import TrainingVisualizer

# Import the environment
from envs import APNetworkEnv

def obs_to_gnn_data(obs, device):
    """
    Converts the observation from APNetworkEnv to a PyG Data object.
    obs["H"]: (N, N) numpy array
    """
    channel_matrix = torch.tensor(obs["H"], dtype=torch.float32).to(device)
    
    # Normalize channel matrix (logic from get_gnn_inputs)
    # norm = torch.norm(channel_matrix, p=2, dim=(0, 1)) # Original uses numpy norm
    # channel_matrix_norm = channel_matrix / norm
    # In original utils.py: channel_matrix_norm = channel_matrix (the norm line is there but then overwritten?)
    # Line 165 in utils.py: channel_matrix_norm = channel_matrix
    # So we stick to that.
    channel_matrix_norm = channel_matrix

    edge_index = channel_matrix_norm.nonzero(as_tuple=False).t()
    edge_attr = channel_matrix_norm[edge_index[0], edge_index[1]]
    edge_attr = edge_attr.to(torch.float)
    
    # Node features x. In original it's zeros (num_links, 1)
    num_links = channel_matrix.shape[0]
    x = torch.zeros((num_links, 1), dtype=torch.float32).to(device)

    data = Data(matrix=channel_matrix, x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def run(building_id=990, b5g=False, num_links=5, num_channels=3, num_layers=5, K=3,
        epochs=100, eps=5e-5, mu_lr=1e-4, synthetic=1, rn=100, rn1=100,
        p0=4, sigma=1e-4, real_time_plotting=False):   

    banda = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str = str(f"{mu_lr:.0e}")

    # --- Data Loading (for the Environment) ---
    # We need an iterator for the environment to get channel matrices
    if synthetic:
        # We use the same synthetic generation but just to get the list of matrices
        _, channel_matrix_tensor = graphs_to_tensor_synthetic(
            num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        # Convert tensor to list of numpy arrays for the iterator
        matrix_list = [m.numpy() for m in channel_matrix_tensor]
    else:
        _, channel_matrix_tensor = graphs_to_tensor(
            train=True, num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        matrix_list = [m.numpy() for m in channel_matrix_tensor]

    # Create an infinite iterator for the environment
    def infinite_iterator(data_list):
        while True:
            random.shuffle(data_list)
            for item in data_list:
                yield item
    
    H_iterator = infinite_iterator(matrix_list)


    mu_k = torch.ones((1, 1), requires_grad=False)

    pmax = num_links * num_channels # Note: This pmax definition in original might be different from Env's Pmax

    # ---- Definición de red ----
    input_dim = 1
    hidden_dim = 1

    # Definí los niveles de potencia discretos
    power_levels = torch.tensor([p0/2, p0])   # ej. 2 niveles
    num_power_levels = len(power_levels)

    # Número total de acciones: no_tx + (canales * niveles de potencia)
    num_actions = 1 + num_channels * num_power_levels

    output_dim = num_actions
    dropout = False
    
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)

    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    probs_values = []   

    # --- Environment Setup ---
    # Note: Env Pmax is relative to P0. Original pmax seems to be total power sum constraint?
    # In original: power_constraint = sum(phi) - pmax.
    # In Env: power_penalty = mu * (avg_power - Pmax).
    # We will trust the Env's internal reward mechanism for now, 
    # BUT the original script minimizes: cost = sum_rate + (power_constr * mu_k)
    # The Env returns reward = rate - power_penalty.
    # So maximizing Env reward is similar to minimizing cost (if we ignore the sign diff).
    # Original: Minimize Cost. Env: Maximize Reward.
    # Cost ~= -Reward.
    
    env = APNetworkEnv(
        n_APs=num_links,
        num_channels=num_channels,
        n_power_levels=num_power_levels,
        P0=p0,
        Pmax=0.7, # Default in Env, adjust if needed to match original 'pmax' logic
        max_steps=50, # Steps per episode
        H_iterator=H_iterator
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = gnn_model.to(device)

    print(f"Starting training on {device}...")

    print(f"Starting training on {device}...")

    # --- Real-time Plotting Setup ---
    if real_time_plotting:
        visualizer = TrainingVisualizer(num_links, num_channels)
    # --------------------------------

    for epoc in range(epochs):
        # In this adapted version, one epoch = one episode (or a fixed number of steps)
        # The original processed the whole dataset in batches.
        # Let's run one full episode per "epoch" iteration to keep structure similar.
        
        obs, _ = env.reset()
        terminated = False
        truncated = False
        
        episode_rewards = []
        episode_loss = []
        episode_probs_accum = [] # To store probs of each step in this episode
        episode_off_probs_accum = [] # To store off probs of each step
        
        # We can accumulate gradients over the episode or update every step.
        # Original updates every batch. Here batch_size=1 (implicitly).
        # Let's update every step to mimic SGD.
        
        steps = 0
        while not (terminated or truncated):
            steps += 1
            
            # 1. Prepare Data
            data = obs_to_gnn_data(obs, device)
            
            # 2. Forward Pass
            # [num_links, num_actions] (since batch=1, we don't have batch dim in output of GNN usually, 
            # but let's check GNN output shape. Original: [batch*num_links, num_actions])
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr) 
            
            # Reshape to [1, num_links, num_actions]
            psi = psi.view(1, num_links, output_dim)
            
            # 3. Action Selection
            probs = torch.softmax(psi, dim=-1) # [1, num_links, num_actions]
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample() # [1, num_links]
            log_p = dist.log_prob(actions) # [1, num_links]
            
            # --- LOGGING DECISIONS AND PROBABILITIES ---
            # Calculate channel probabilities
            # probs shape: [1, num_links, 1 + num_channels * num_power_levels]
            # Action 0 is OFF.
            # Actions 1..num_power_levels -> Channel 1
            # Actions num_power_levels+1..2*num_power_levels -> Channel 2
            
            # Extract probabilities for active actions (excluding action 0)
            active_probs = probs[0, :, 1:] # [num_links, num_channels * num_power_levels]
            
            # Reshape to [num_links, num_channels, num_power_levels]
            probs_reshaped = active_probs.view(num_links, num_channels, num_power_levels)
            channel_probs = probs_reshaped.sum(dim=2) # [num_links, num_channels]
            
            # Also get prob of being OFF
            off_probs = probs[0, :, 0] # [num_links]
            
            # Store for epoch average
            episode_probs_accum.append(channel_probs.detach().cpu().numpy()) # [num_links, num_channels]
            episode_off_probs_accum.append(off_probs.detach().cpu().numpy()) # [num_links]
            
            print(f"\nStep {steps} Decisions & Probabilities:")
            for i in range(num_links):
                # Decode action
                a = actions[0, i].item()
                if a == 0:
                    decision_str = "OFF"
                else:
                    a_adj = a - 1
                    c = a_adj // num_power_levels + 1
                    p_idx = a_adj % num_power_levels
                    p_val = power_levels[p_idx].item()
                    decision_str = f"Ch {c}, P {p_val:.2f}"
                
                # Format channel probs
                c_probs_str = ", ".join([f"Ch{c+1}: {p:.2f}" for c, p in enumerate(channel_probs[i].tolist())])
                print(f"  AP {i}: Decision: {decision_str} | Off Prob: {off_probs[i]:.2f} | {c_probs_str}")
            # -------------------------------------------
            
            # Sum log probs for the "joint" action of all APs (centralized training)
            log_p_sum = log_p.sum(dim=1).unsqueeze(-1) # [1, 1]
            
            # 4. Step Environment
            action_numpy = actions.cpu().numpy()[0] # [num_links]
            next_obs, reward, terminated, truncated, info = env.step(action_numpy)
            
            # 5. Calculate Loss
            # Original: loss = cost * log_p_sum
            # cost = sum_rate + constraint.
            # Env reward = rate - penalty.
            # So Cost ~= -Reward.
            # We want to minimize Loss.
            # REINFORCE: minimize - (Reward * log_p).
            # If we define Cost = -Reward, then minimize Cost * log_p is WRONG sign?
            # Wait.
            # Maximize J = E[Reward]. Gradient is E[grad(log_p) * Reward].
            # Loss to minimize = - log_p * Reward.
            # Original script:
            # cost = sum_rate + ... (This is actually "negative rate" in original? 
            # Let's check objective_function in utils.py)
            # utils.py: sum_rate = -torch.sum(rates). So "sum_rate" is NEGATIVE.
            # So "cost" is indeed a cost (lower is better).
            # Env reward is POSITIVE (rate - penalty).
            # So Cost = -Reward is correct.
            # Loss = Cost * log_p_sum = (-Reward) * log_p_sum.
            # Minimizing this pushes log_p_sum up when Reward is high (Cost is low/negative).
            # Correct.
            
            cost = -reward
            loss = cost * log_p_sum
            
            # 6. Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Logging
            episode_rewards.append(reward)
            episode_loss.append(loss.item())
            
            obs = next_obs
            
            # Optional: Store metrics for plotting (taking mean of this step)
            # This might be too noisy per step, maybe average per episode?
            # Original stored every 10 batches.
        
        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean(episode_loss)
        
        print(f"Epoch {epoc}: Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")
        
        objective_function_values.append(avg_reward) # Plotting positive reward
        loss_values.append(avg_loss)

        # --- Update Plot using Visualizer ---
        if real_time_plotting:
            avg_probs_episode = np.mean(np.array(episode_probs_accum), axis=0) # [num_links, num_channels]
            avg_off_probs_episode = np.mean(np.array(episode_off_probs_accum), axis=0) # [num_links]
            
            visualizer.update(avg_reward, avg_loss, avg_probs_episode, avg_off_probs_episode)
        # ------------------------------------
        
        # power_constraint and mu are handled inside env, we could extract from info if needed
        # For now let's just track main objective.

    # Save results
    # We reuse plot_results but we might need to adapt inputs as we don't have all the exact lists
    # Or we just save the data.
    
    print("Training finished.")
    
    if real_time_plotting:
        visualizer.close()
    
    # Save model
    torch.save(gnn_model.state_dict(), 'gnn_model_weights_adapted.pth')
    
    # Simple plot
    plt.figure()
    plt.plot(objective_function_values)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Training Reward over Epochs')
    plt.savefig('training_reward.png')
    print("Saved training_reward.png")
        



if __name__ == '__main__':
    import argparse

    rn = 267309
    rn1 = 502321
    torch.manual_seed(rn)
    np.random.seed(rn1) 
    random.seed(rn1)

    parser = argparse.ArgumentParser(description= 'System configuration')

    parser.add_argument('--building_id', type=int, default=990)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=5)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    # parser.add_argument('--batch_size', type=int, default=64) # Not used
    parser.add_argument('--eps', type=float, default=5e-5)
    parser.add_argument('--mu_lr', type=float, default=5e-5)
    parser.add_argument('--synthetic', type=int, default=1) # Default to synthetic for test

    args = parser.parse_args()
    
    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, 
        num_channels=args.num_channels, num_layers=args.num_layers, K=args.k, 
        epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, rn=rn, rn1=rn1,
        real_time_plotting=True)
