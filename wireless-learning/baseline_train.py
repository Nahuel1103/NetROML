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
from utils import power_constraint
from utils import get_rates
from utils import objective_function
from utils import mu_update

from utils import graphs_to_tensor_synthetic

def run(building_id=990, b5g=False, num_channels=5, num_layers=5, K=3, batch_size=64, epocs=100, eps=5e-5, mu_lr=1e-4, baseline=1, synthetic=1, rn=100, rn1=100):   

    banda = ['2_4', '5']
    if synthetic:
        x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(num_channels,num_features=1, b5g=b5g, building_id=building_id)
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset[:7000], batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        x_tensor, channel_matrix_tensor = graphs_to_tensor(train=False, num_channels=num_channels, num_features=1, b5g=b5g, building_id=building_id)
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    mu_k = torch.ones((1,1), requires_grad = False)
    epocs = epocs

    pmax = num_channels
    p0 = 4

    sigma = 1e-4

    input_dim = 1
    hidden_dim = 1
    output_dim = 1
    num_layers = num_layers
    dropout = False
    K = K
    
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)

    optimizer = optim.Adam(gnn_model.parameters(), lr= mu_lr)

    for name, param in gnn_model.named_parameters():
        param.requires_grad = True
        if param.requires_grad:
            print(name, param.data)

    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    normalized_psi_values = []
    for epoc in range(epocs):
        print("Epoc number: {}".format(epoc))
        for batch_idx, data in enumerate(dataloader):
            
            channel_matrix_batch = data.matrix
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_channels, num_channels)
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            psi = psi.squeeze(-1)
            psi = psi.view(batch_size, -1)
            psi = psi.unsqueeze(-1)
            
            normalized_psi = torch.sigmoid(psi)*(0.99 - 0.01) + 0.01
            if (baseline==1):
                normalized_psi = torch.ones((batch_size, num_channels,1)) / 2
            elif (baseline==2):
                normalized_psi = torch.ones((batch_size, num_channels,1)) / num_channels

            normalized_psi_values.append(normalized_psi[0,:,:].squeeze(-1).detach().numpy())


            normalized_phi = torch.bernoulli(normalized_psi)
            log_p = normalized_phi * torch.log(normalized_psi) + (1 - normalized_phi) * torch.log(1 - normalized_psi)
            log_p_sum = torch.sum(log_p, dim=1)
            if (baseline==1):
                normalized_psi = pmax/(p0*num_channels) * torch.ones((64,num_channels,1))
                normalized_phi = torch.bernoulli(normalized_psi)
                phi = normalized_phi * p0
            elif (baseline==2):
                phi = normalized_phi * pmax

            power_constr = power_constraint(phi, pmax)
            power_constr_mean = torch.mean(power_constr, dim = 0)
            
            rates = get_rates(phi, channel_matrix_batch, sigma)

            sum_rate = objective_function(rates)

            sum_rate_mean = torch.mean(sum_rate, dim = 0)
            
            mu_k = mu_update(mu_k, power_constr, eps)
            
            cost = sum_rate + (power_constr * mu_k)

            loss = cost * log_p_sum
            loss_mean = torch.mean(loss, dim = 0)

            optimizer.zero_grad()

            if batch_idx%10 == 0:
                power_constraint_values.append(power_constr_mean.detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().numpy())
    
    path = plot_results(building_id=building_id, b5g=b5g, normalized_psi=normalized_psi, normalized_psi_values=normalized_psi_values, num_layers=num_layers, K=K, batch_size=batch_size, epocs=epocs, rn=rn, rn1=rn1, eps=eps,
                    objective_function_values=objective_function_values, power_constraint_values=power_constraint_values,
                    loss_values=loss_values, mu_k_values=mu_k_values, baseline=baseline, synthetic=synthetic)
    
    #path = '../results/'+str(banda[b5g])+'_'+str(building_id)+'/torch_results/'
    file_name = path + 'baseline'+str(baseline)+'_'+str(epocs)+'.pkl'
    with open(file_name, 'wb') as archivo:
        pickle.dump(objective_function_values, archivo)
        
if __name__ == '__main__':
    import argparse

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    rn = 267309
    rn1 = 502321
    torch.manual_seed(rn)
    np.random.seed(rn1) 

    parser = argparse.ArgumentParser()

    parser.add_argument('--building_id', type=int, default=990)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_channels', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epocs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=5e-5)
    parser.add_argument('--mu_lr', type=float, default=5e-5)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--baseline', type=int, default=1)

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
    print(f'baseline: {args.baseline}')

    

    run(building_id=args.building_id, b5g=args.b5g, num_channels=args.num_channels, num_layers=args.num_layers, K=args.k, batch_size=args.batch_size, epocs=args.epocs, eps=args.eps, mu_lr=args.mu_lr, baseline=args.baseline, synthetic=args.synthetic, rn=rn, rn1=rn1)
    print(rn)
    print(rn1)

