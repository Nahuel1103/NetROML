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
        batch_size=64, epochs=100, eps=5e-5, mu_lr=5e-4, synthetic=1, rn=100, rn1=100,
         sigma=1e-4,max_antenna_power_dbm=6):   

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
            train=False, num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
    pmax_per_ap = max_antenna_power_mw*0.6* torch.ones((num_links,))
    mu_k =  torch.ones((num_links,), requires_grad=False)  


    # ---- Definición de red ----
    input_dim = 1
    hidden_dim = 1


    # Definí los niveles de potencia discretos
    power_levels = torch.tensor([ max_antenna_power_mw/2, max_antenna_power_mw])  # Incluimos 0 = no transmitir
    num_power_levels = len(power_levels)

    output_dim = 2 + num_channels + num_power_levels

    dropout = True
    
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)

    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    tx_policy_values = []      # Cada elemento: [P(no_tx), P(tx)]
    channel_policy_values = [] # Cada elemento: [P(canal_0), P(canal_1), P(canal_2)]
    power_policy_values = []   # Cada elemento: [P(p0/2), P(p0)] 
    
    final_tx_probs = None
    final_channel_probs = None
    final_power_probs = None

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
            action_logits = psi                                                    # [batch, links, output_dim]
            action_probs = torch.softmax(action_logits, dim=-1)                    # [batch, links, output_dim]
            action_dist = torch.distributions.Categorical(probs=action_probs)
            actions = action_dist.sample()                                         # [batch, links] - valores 0..(output_dim-1)
            log_p_actions = action_dist.log_prob(actions)                          # [batch, links]

            # Mascara de transmisión (acción != 0)
            transmit_mask = (actions != 0)
            # Log probability total por muestra (sumamos sobre links)
            log_p_sum = log_p_actions.sum(dim=1).unsqueeze(-1)                     # [batch, 1]

            # Construir phi (potencias por canal)
            phi = torch.zeros(batch_size, num_links, num_channels)

            # Solo asignar potencia si decide transmitir
            transmit_indices = torch.where(transmit_mask)
            if len(transmit_indices[0]) > 0:
                batch_idx_tx = transmit_indices[0]
                link_idx_tx = transmit_indices[1]
                selected_actions = actions[batch_idx_tx, link_idx_tx]             # valores 1..(C*P)
                # convertir a canal y potencia
                selected_minus1 = selected_actions - 1
                selected_channels = (selected_minus1 // num_power_levels).long()
                selected_powers_idx = (selected_minus1 % num_power_levels).long()
                # Obtener valores reales de potencia (ej. en mW)
                selected_powers_values = power_levels[selected_powers_idx].to(phi.dtype)
                # Asignar phi
                phi[batch_idx_tx, link_idx_tx, selected_channels] = selected_powers_values

            # Para logging de políticas: obtener marginales a partir de action_probs
            # Probabilidad de no transmitir
            p_no_tx = action_probs[:,:,0]                                           # [batch, links]
            # Probabilidades de las combinaciones (reshape a [batch,links,num_channels,num_power_levels])
            combos = action_probs[:,:,1:]
            combos = combos.view(batch_size, num_links, num_channels, num_power_levels)
            # Marginal por canal (sumando sobre niveles de potencia)
            channel_marginal = combos.sum(dim=-1)                                   # [batch, links, num_channels]
            # Marginal por potencia (sumando sobre canales)
            power_marginal = combos.sum(dim=2)                                      # [batch, links, num_power_levels]

            # Guardar promedios para monitoreo
            tx_policy_values.append(p_no_tx.mean(dim=[0,1]).detach().numpy())      # media de P(no_tx)
            channel_policy_values.append(channel_marginal.mean(dim=[0,1]).detach().numpy())
            power_policy_values.append(power_marginal.mean(dim=[0,1]).detach().numpy())
            
            final_tx_probs = p_no_tx
            final_channel_probs = channel_marginal
            final_power_probs = power_marginal

            power_constr_per_ap = power_constraint_per_ap(phi, pmax_per_ap)  # [batch_size, num_links]
            power_constr_per_ap_mean = torch.mean(power_constr_per_ap, dim=(0,1))  
            rates = nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=max_antenna_power_mw)

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
                tx_policy_values.append(p_no_tx.mean(dim=[0,1]).detach().numpy())
                channel_policy_values.append(channel_marginal.mean(dim=[0,1]).detach().numpy()) 
                power_policy_values.append(power_marginal.mean(dim=[0,1]).detach().numpy())
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
            tx_probs=final_tx_probs,
            channel_probs=final_channel_probs, 
            power_probs=final_power_probs,
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

    file_name = path + 'objective_function_values_val_' + str(epochs) + '.pkl'
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

    parser.add_argument('--building_id', type=int, default=856)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=20)
    parser.add_argument('--num_channels', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--mu_lr', type=float, default=1e-3)
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