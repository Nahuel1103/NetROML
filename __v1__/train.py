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
from torch.distributions import Categorical

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TAGConv
from torch_geometric.nn import GCNConv
import os
import matplotlib.pyplot as plt

import scipy.io

from plot_results import plot_results
from gnn import GNN
from utils import mu_update
# Env import - Importing our custom Gymnasium environment
from env_v1 import WirelessEnv

def run(building_id=990, b5g=False, num_links=5, num_layers=5, K=3, batch_size=64, epochs=100, eps=5e-5, mu_lr=1e-4, synthetic=1, rn=100, rn1=100):   
    """
    Main training execution function.
    
    Args:
        building_id (int): ID of the building.
        b5g (bool): Use 5G band if True.
        num_channels (int): Number of channels/nodes in graph.
        num_layers (int): Number of GNN layers.
        K (int): Order of Chebyshev polynomials / hops in TAGConv.
        batch_size (int): Size of training batches.
        epochs (int): Number of training epochs.
        eps (float): Step size for Lagrangian multiplier update.
        mu_lr (float): Learning rate for the Optimizer (Adam).
        synthetic (int): 1 for synthetic data, 0 for real data.
        rn, rn1 (int): Random seeds.
    """

    banda = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str= str(f"{mu_lr:.0e}")

    # 1. Initialize Environment
    env = WirelessEnv(building_id=building_id, b5g=b5g, num_links=num_links, num_features=1, synthetic=synthetic, train=True)
    
    # 2. Setup Data Loader
    dataloader = DataLoader(env.dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize Lagrangian Multiplier (mu_k)
    # Used to enforce power constraints via primal-dual method
    mu_k = torch.ones((1,1), requires_grad = False)
    epochs = epochs

    # Parameters for Power Calculation
    max_antenna_power_dbm = 6
    max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
    pmax_per_ap = max_antenna_power_mw*0.8* torch.ones((num_links,))

    num_channels = 3

    sigma = 1e-4 # Noise

    # GNN Architecture Dimensions
    input_dim = 1
    hidden_dim = 1
    output_dim = num_channels + 1  # [no_tx, canal_0, canal_1, canal_2, ...]
    num_layers = num_layers
    dropout = False # Dropout not used
    K = K # Filter size

    # 3. Model Initialization
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)

    # 4. Optimizer Setup
    optimizer = optim.Adam(gnn_model.parameters(), lr= mu_lr)

    # Enable gradients for all model parameters and print initial values
    for name, param in gnn_model.named_parameters():
        param.requires_grad = True
        if param.requires_grad:
            print(name, param.data)

    # Lists to store metrics for plotting
    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    probs_values = []
    power_values = [] 
    
    # 5. Training Loop
    for epoc in range(epochs):
        print("Epoc number: {}".format(epoc))
        
        # Iterate over batches from the dataloader
        for batch_idx, data in enumerate(dataloader):
            
            # --- PREPARATION ---
            # Get channel matrix for the batch from Data object
            channel_matrix_batch = data.matrix
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_links, num_links)

            # --- GNN FORWARD PASS ---
            # Pass graph data (features, connectivity) through GNN
            logits = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            
            # Reshape logits to (Batch_Size, Links, Channels+1)
            psi = logits.view(batch_size, num_links, num_channels+1)
            # Softmax to get probabilities
            probs = F.softmax(psi, dim=-1)  
            # Sample actions from the distribution
            m = Categorical(probs=probs)  
            actions = m.sample()  
            # Get log probabilities
            log_p = m.log_prob(actions)     
            log_p_sum = log_p.sum(dim=1).unsqueeze(-1) 
            
            phi = torch.zeros(batch_size, num_links, num_channels, device=probs.device)
            active_mask = (actions > 0)
            active_channels = actions[active_mask] - 1
            phi[active_mask, active_channels] = max_antenna_power_mw 

            # --- ENVIRONMENT STEP ---
            # Pass the action 'phi' to the Gymnasium environment environment to get feedback.
            # We inject the current batch data into the env to ensure it calculates physics for THIS batch.
            env.data = data 
            
            # Helper logic inside env.step will compute 'rates' and 'reward' based on 'phi' and 'env.data.matrix'
            obs, reward, terminated, truncated, info = env.step(phi)
            
            # --- METRIC EXTRACTION ---
            # Extract info from environment return
            power_constr_per_ap = info['power_constr'] # Constraint violation
            
            # Retrieve sum_rate (negative of reward, since reward = +Rate)
            # Note: The 'cost' for minimization usually includes negative Utility.
            sum_rate = -reward
            
            # Calculate means for logging
            power_constr_mean = torch.mean(power_constr_per_ap, dim = 0)
            sum_rate_mean = torch.mean(sum_rate, dim = 0)
            
            # --- LAGRANGIAN UPDATE ---
            # Update mu_k based on constraint violation
            mu_k = mu_update(mu_k, power_constr_per_ap, eps)
            
            # --- LOSS CALCULATION ---
            # Cost = Objective (neg Rate) + Punishment (Constraint Violation * mu_k)
            # mu_k shape (1,1) broadcasts with power_constr_per_ap (batch, links)
            penalty_per_ap = power_constr_per_ap * mu_k  
            total_penalty = penalty_per_ap.sum(dim=1).unsqueeze(-1)   

            # Unsqueeze sum_rate to (batch, 1) to match total_penalty shape and avoid broadcasting error
            cost = sum_rate.unsqueeze(-1) + total_penalty 

            # Policy Gradient Loss: Cost * Log_Probability
            # This is the standard REINFORCE update rule (weighted log-likelihood)
            loss = cost * log_p_sum
            loss_mean = torch.mean(loss, dim = 0)
        
            # --- BACKPROPAGATION ---
            optimizer.zero_grad()                                                   # 1. Limpia gradientes primero
            loss_mean.backward()                                                    # 2. Calcula gradientes
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=5.0)    # 3. Clip gradientes
            optimizer.step()                                                        # 4. Actualiza parámetros
            
            # --- LOGGING ---
            if batch_idx%10 == 0:
                probs_mean_batch = probs.mean(dim=0)  
                probs_values.append(probs_mean_batch.detach().numpy())
                
                power_constraint_values.append(power_constr_mean.detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().numpy())

    # Print final weights
    for name, param in gnn_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    
    # 6. Plotting Results
    path = plot_results(building_id=building_id, b5g=b5g, normalized_psi=probs, normalized_psi_values=probs_values, num_layers=num_layers, K=K, batch_size=batch_size, epochs=epochs, rn=rn, rn1=rn1, eps=eps, mu_lr=mu_lr, objective_function_values=objective_function_values, power_constraint_values=power_constraint_values, loss_values=loss_values, mu_k_values=mu_k_values, baseline=False, train=True, synthetic=synthetic)
    
    # Save raw objective values
    file_name = path + 'objective_function_values_train_' + str(epochs) + '.pkl'
    with open(file_name, 'wb') as archivo:
        pickle.dump(objective_function_values, archivo)

    # Save trained model weights
    torch.save(gnn_model.state_dict(), path + 'gnn_model_weights.pth')
        
if __name__ == '__main__':
    import argparse

    # Setup Random Seeds
    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    rn = 267309
    rn1 = 502321
    torch.manual_seed(rn)
    np.random.seed(rn1) 

    # Argument Parser for Command Line Execution
    parser = argparse.ArgumentParser(description= 'System configuration')

    parser.add_argument('--building_id', type=int, default=990)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eps', type=float, default=5e-4)
    parser.add_argument('--mu_lr', type=float, default=5e-4)
    parser.add_argument('--synthetic', type=int, default=1)
    
    args = parser.parse_args()
    
    print(f'building_id: {args.building_id}')
    print(f'b5g: {args.b5g}')
    print(f'num_links: {args.num_links}')
    print(f'num_layers: {args.num_layers}')
    print(f'k: {args.k}')
    print(f'epochs: {args.epochs}')
    print(f'batch_size: {args.batch_size}')
    print(f'eps: {args.eps}')
    print(f'mu_lr: {args.mu_lr}')
    print(f'synthetic: {args.synthetic}')
    
    # Run the training loop
    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, num_layers=args.num_layers, K=args.k, batch_size=args.batch_size, epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, rn=rn, rn1=rn1)
    print('Seeds: {} and {}'.format(rn, rn1))
