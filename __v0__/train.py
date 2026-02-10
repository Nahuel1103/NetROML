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

from plot_results import plot_results
from gnn import GNN
from utils import mu_update
# Env import - Importing our custom Gymnasium environment
from env_v0 import WirelessEnv

def run(building_id=990, b5g=False, num_channels=5, num_layers=5, K=3, batch_size=1, epocs=100, eps=5e-5, mu_lr=1e-4, synthetic=1, rn=100, rn1=100):   
    """
    Main training execution function.
    
    Args:
        building_id (int): ID of the building.
        b5g (bool): Use 5G band if True.
        num_channels (int): Number of channels/nodes in graph.
        num_layers (int): Number of GNN layers.
        K (int): Order of Chebyshev polynomials / hops in TAGConv.
        batch_size (int): Size of training batches.
        epocs (int): Number of training epochs.
        eps (float): Step size for Lagrangian multiplier update.
        mu_lr (float): Learning rate for the Optimizer (Adam).
        synthetic (int): 1 for synthetic data, 0 for real data.
        rn, rn1 (int): Random seeds.
    """

    banda = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str= str(f"{mu_lr:.0e}")

    # 1. Initialize Environment
    # We create the WirelessEnv instance which prepares the dataset.
    env = WirelessEnv(building_id=building_id, b5g=b5g, num_channels=num_channels, num_features=1, synthetic=synthetic, train=True)
    
    # 2. Setup Data Loader
    # We use the dataset from the environment to create a PyTorch Geometric DataLoader.
    # This allows efficient batching of graph data.
    dataloader = DataLoader(env.dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize Lagrangian Multiplier (mu_k)
    # Used to enforce power constraints via primal-dual method
    mu_k = torch.ones((1,1), requires_grad = False)
    epocs = epocs

    # Parameters for Power Calculation
    pmax = num_channels # Max power budget
    p0 = 4              # Base power unit

    sigma = 1e-4 # Noise

    # GNN Architecture Dimensions
    input_dim = 1
    hidden_dim = 1
    output_dim = 1
    num_layers = num_layers
    dropout = False # Dropout not used
    K = K # Filter size

    # 3. Model Initialization
    # Instantiate the GNN model
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)

    # 4. Optimizer Setup
    # Use Adam optimizer for GNN weights
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
    normalized_psi_values = []
    
    # 5. Training Loop
    for epoc in range(epocs):
        print("Epoc number: {}".format(epoc))
        
        # Iterate over batches from the dataloader
        for batch_idx, data in enumerate(dataloader):
            
            # --- PREPARATION ---
            # Get channel matrix for the batch from Data object
            channel_matrix_batch = data.matrix
            # Reshape to (Batch_Size, N, N) for matrix operations
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_channels, num_channels)

            # --- GNN FORWARD PASS ---
            # Pass graph data (features, connectivity) through GNN
            # psi: Output logits/features from GNN, shape (Batch*Nodes, 1) or similar before view
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            
            # Reshape psi to (Batch_Size, Nodes, 1)
            psi = psi.squeeze(-1)
            psi = psi.view(batch_size, -1)
            psi = psi.unsqueeze(-1)
            
            # Normalize GNN output to probability range [0, 1] using Tanh + scaling
            # normalized_psi represents the probability of activating a node/channel
            normalized_psi = (torch.tanh(psi)*(0.99 - 0.01) + 1)/2
            
            # Store first sample's policy for plotting
            normalized_psi_values.append(normalized_psi[0,:,:].squeeze(-1).detach().numpy())

            # --- ACTION SELECTION (Stochastic Policy) ---
            # Sample binary actions from Bernoulli distribution parameterized by normalized_psi
            normalized_phi = torch.bernoulli(normalized_psi)
            
            # Compute Log Probabilities for REINFORCE / Policy Gradient
            # log_p = phi * log(prob) + (1-phi) * log(1-prob)
            log_p = normalized_phi * torch.log(normalized_psi) + (1 - normalized_phi) * torch.log(1 - normalized_psi)
            # Sum log probs over all decisions in the graph to get total log prob for the trajectory/graph
            log_p_sum = torch.sum(log_p, dim=1)
            
            # Scale binary action by power unit p0 to get actual Power Allocation
            phi = normalized_phi * p0

            # --- ENVIRONMENT STEP ---
            # Pass the action 'phi' to the Gymnasium environment environment to get feedback.
            # We inject the current batch data into the env to ensure it calculates physics for THIS batch.
            env.data = data 
            
            # Helper logic inside env.step will compute 'rates' and 'reward' based on 'phi' and 'env.data.matrix'
            obs, reward, terminated, truncated, info = env.step(phi)
            
            # --- METRIC EXTRACTION ---
            # Extract info from environment return
            power_constr = info['power_constr'] # Constraint violation
            
            # Retrieve sum_rate (negative of reward, since reward = +Rate)
            # Note: The 'cost' for minimization usually includes negative Utility.
            sum_rate = -reward
            
            # Calculate means for logging
            power_constr_mean = torch.mean(power_constr, dim = 0)
            sum_rate_mean = torch.mean(sum_rate, dim = 0)
            
            # --- LAGRANGIAN UPDATE ---
            # Update mu_k based on constraint violation
            mu_k = mu_update(mu_k, power_constr, eps)
            
            # --- LOSS CALCULATION ---
            # Cost = Objective (neg Rate) + Punishment (Constraint Violation * mu_k)
            cost = sum_rate + (power_constr * mu_k)

            # Policy Gradient Loss: Cost * Log_Probability
            # This is the standard REINFORCE update rule (weighted log-likelihood)
            loss = cost * log_p_sum
            loss_mean = torch.mean(loss, dim = 0)
        
            # --- BACKPROPAGATION ---
            loss_mean.backward()
            optimizer.step()
            optimizer.zero_grad() # Clear gradients for next step

            # --- LOGGING ---
            if batch_idx%10 == 0:
                power_constraint_values.append(power_constr_mean.detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy()) # Log Positive Rate
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().numpy())

    # Print final weights
    for name, param in gnn_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    
    # 6. Plotting Results
    path = plot_results(building_id=building_id, b5g=b5g, normalized_psi=normalized_psi, normalized_psi_values=normalized_psi_values, num_layers=num_layers, K=K, batch_size=batch_size, epocs=epocs, rn=rn, rn1=rn1, eps=eps, mu_lr=mu_lr,
                    objective_function_values=objective_function_values, power_constraint_values=power_constraint_values,
                    loss_values=loss_values, mu_k_values=mu_k_values, baseline=False, synthetic=synthetic, train=True)
    
    # Save raw objective values
    file_name = path + 'objective_function_values_train_' + str(epocs) + '.pkl'
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
    parser.add_argument('--num_channels', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epocs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=5e-4)
    parser.add_argument('--mu_lr', type=float, default=5e-4)
    parser.add_argument('--synthetic', type=int, default=1)
    
    args = parser.parse_args()
    
    print(f'building_id: {args.building_id}')
    print(f'b5g: {args.b5g}')
    print(f'num_channels: {args.num_channels}')
    print(f'num_layers: {args.num_layers}')
    print(f'k: {args.k}')
    print(f'epocs: {args.epocs}')
    print(f'batch_size: {args.batch_size}')
    print(f'eps: {args.eps}')
    print(f'mu_lr: {args.mu_lr}')
    print(f'synthetic: {args.synthetic}')
    
    # Run the training loop
    run(building_id=args.building_id, b5g=args.b5g, num_channels=args.num_channels, num_layers=args.num_layers, K=args.k, batch_size=args.batch_size, epocs=args.epocs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, rn=rn, rn1=rn1)
    print('Seeds: {} and {}'.format(rn, rn1))
