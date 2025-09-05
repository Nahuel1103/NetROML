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
from utils import power_constraint_per_ap
from utils import objective_function
from utils import mu_update
# from utils import channel_constraint
from utils import nuevo_get_rates

from utils import graphs_to_tensor_synthetic


def run(building_id=990, b5g=False, num_links=5, num_channels=3, num_layers=5, K=3,
        batch_size=64, epochs=100, eps=5e-5, mu_lr=1e-4, synthetic=1, rn=100, rn1=100,
        p0=11, sigma=1e-4):   

    banda = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str = str(f"{mu_lr:.0e}")

    if synthetic:
        x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
            num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset[:7000], batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        x_tensor, channel_matrix_tensor = graphs_to_tensor(
            train=True, num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    mu_k = torch.ones(num_links, requires_grad=False)

    pmax = torch.ones(num_links) * 4*0.8 # [num_links]

    # ---- Definición de red ----
    input_dim = 1
    hidden_dim = 1


    # Definí los niveles de potencia discretos
    power_levels = torch.tensor([0, p0/2, p0])  # Incluimos 0 = no transmitir
    num_power_levels = len(power_levels)

    output_dim = 2 + num_channels + num_power_levels

    dropout = False
    
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)

    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    tx_policy_values = []      # Cada elemento: [P(no_tx), P(tx)]
    channel_policy_values = [] # Cada elemento: [P(canal_0), P(canal_1), P(canal_2)]
    power_policy_values = []   # Cada elemento: [P(p0/2), P(p0)] 

    for epoc in range(epochs):
        print("Epoch number: {}".format(epoc))
        for batch_idx, data in enumerate(dataloader):
    
            channel_matrix_batch = data.matrix
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_links, num_links) # [64, 5, 5]
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)   # [batch*num_links, num_actions]
            psi = psi.view(batch_size, num_links, output_dim)                  # [batch, num_links, num_actions]
          
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            psi = psi.view(batch_size, num_links, output_dim)

            # Dividir las salidas en 3 decisiones
            tx_decision_logits = psi[:, :, :2]                                    # [batch, links, 2]
            channel_decision_logits = psi[:, :, 2:2+num_channels]                # [batch, links, num_channels] 
            power_decision_logits = psi[:, :, 2+num_channels:]                   # [batch, links, num_power_levels]

            # Probabilidades independientes para cada decisión
            tx_probs = torch.softmax(tx_decision_logits, dim=-1)                 # [batch, links, 2]
            channel_probs = torch.softmax(channel_decision_logits, dim=-1)       # [batch, links, num_channels]
            power_probs = torch.softmax(power_decision_logits, dim=-1)           # [batch, links, num_power_levels]

            # Muestrear cada decisión independientemente
            tx_dist = torch.distributions.Categorical(probs=tx_probs)
            channel_dist = torch.distributions.Categorical(probs=channel_probs)
            power_dist = torch.distributions.Categorical(probs=power_probs)

            tx_actions = tx_dist.sample()          # [batch, links] - 0: no tx, 1: tx
            channel_actions = channel_dist.sample() # [batch, links] - índice del canal
            power_actions = power_dist.sample()     # [batch, links] - índice del nivel de potencia

            # Log probabilities para el gradiente
            log_p_tx = tx_dist.log_prob(tx_actions)
            log_p_channel = channel_dist.log_prob(channel_actions) 
            log_p_power = power_dist.log_prob(power_actions)

            # Solo contar log_prob si realmente transmite
            transmit_mask = (tx_actions == 1)
            log_p_total = log_p_tx.clone()
            log_p_total[transmit_mask] += log_p_channel[transmit_mask] + log_p_power[transmit_mask]
            log_p_sum = log_p_total.sum(dim=1).unsqueeze(-1)

            # Construir phi
            phi = torch.zeros(batch_size, num_links, num_channels, device=psi.device)

            # Solo asignar potencia si decide transmitir
            transmit_indices = torch.where(transmit_mask)
            if len(transmit_indices[0]) > 0:
                batch_idx_tx = transmit_indices[0]
                link_idx_tx = transmit_indices[1]
                selected_channels = channel_actions[batch_idx_tx, link_idx_tx]
                selected_powers = power_actions[batch_idx_tx, link_idx_tx]
                
                phi[batch_idx_tx, link_idx_tx, selected_channels] = power_levels.to(phi.device)[selected_powers]

            power_constr_per_ap = power_constraint_per_ap(phi, pmax)  # [batch_size, num_links]
            power_constr_per_ap_mean = torch.mean(power_constr_per_ap, dim=(0,1))  
            rates = nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=p0)

            sum_rate = -objective_function(rates).unsqueeze(-1) 
            sum_rate_mean = torch.mean(sum_rate, dim=0)

            mu_k = mu_update(mu_k, power_constr_per_ap, eps) 

            penalty_per_ap = power_constr_per_ap * mu_k.unsqueeze(0)  
            total_penalty = penalty_per_ap.sum(dim=1).unsqueeze(-1)   

            cost = sum_rate + total_penalty  
            
            loss = cost * log_p_sum        
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()


            if batch_idx % 10 == 0:
                tx_policy_values.append(tx_probs.mean(dim=[0,1]).detach().numpy())
                channel_policy_values.append(channel_probs.mean(dim=[0,1]).detach().numpy()) 
                power_policy_values.append(power_probs.mean(dim=[0,1]).detach().numpy())
                power_constraint_values.append(power_constr_per_ap_mean.detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().numpy())


    for name, param in gnn_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    path = plot_results(
            building_id=building_id,
            b5g=b5g,
            tx_probs=tx_probs,
            channel_probs=channel_probs, 
            power_probs=power_probs,
            tx_policy_values=tx_policy_values,
            channel_policy_values=channel_policy_values,
            power_policy_values=power_policy_values,
            num_layers=num_layers,
            K=K,
            batch_size=batch_size,
            epochs=epochs,
            rn=rn,
            rn1=rn1,
            eps=eps,
            mu_lr=mu_lr,
            objective_function_values=objective_function_values,
            power_constraint_values=power_constraint_values,
            loss_values=loss_values,
            mu_k_values=mu_k_values,
            train=True,
            synthetic=synthetic,
            num_channels=num_channels,
            num_power_levels=num_power_levels
        )

    file_name = path + 'objective_function_values_train_' + str(epochs) + '.pkl'
    with open(file_name, 'wb') as archivo:
        pickle.dump(objective_function_values, archivo)

    # save trained gnn weights in .pth
    torch.save(gnn_model.state_dict(), path + 'gnn_model_weights.pth')
        
if __name__ == '__main__':
    import argparse

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    rn = 267309
    rn1 = 502321
    torch.manual_seed(rn)
    np.random.seed(rn1) 

    parser = argparse.ArgumentParser(description= 'System configuration')

    parser.add_argument('--building_id', type=int, default=990)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=5)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=5e-5)
    parser.add_argument('--mu_lr', type=float, default=5e-5)
    parser.add_argument('--synthetic', type=int, default=0)

    args = parser.parse_args()
    
    print(f'building_id: {args.building_id}')
    print(f'b5g: {args.b5g}')
    print(f'num_links: {args.num_links}')
    print(f'num_channels: {args.num_channels}')
    print(f'num_layers: {args.num_layers}')
    print(f'k: {args.k}')
    print(f'epochs: {args.epochs}')
    print(f'batch_size: {args.batch_size}')
    print(f'eps: {args.eps}')
    print(f'mu_lr: {args.mu_lr}')
    print(f'synthetic: {args.synthetic}')

    

    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, num_channels=args.num_channels, num_layers=args.num_layers, K=args.k, batch_size=args.batch_size, epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, rn=rn, rn1=rn1)
    
    print('Seeds: {} and {}'.format(rn, rn1))