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
from utils import objective_function
from utils import mu_update
# from utils import channel_constraint
from utils import nuevo_get_rates

from utils import graphs_to_tensor_synthetic


def run(building_id=990, b5g=False, num_links=5, num_channels=3, num_layers=5, K=3,
        batch_size=64, epochs=100, eps=5e-5, mu_lr=1e-4, synthetic=1, rn=100, rn1=100,
        p0=4, sigma=1e-4):   

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

    mu_k = torch.ones((1, 1), requires_grad=False)

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
    # optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)
    optimizer = optim.Adam(gnn_model.parameters(), lr=1e-2)

    objective_function_values = []
    loss_values = []
    probs_values = []   
    power_constraint_values = []
    mu_k_values = []


    for epoc in range(epochs):
        print("Epoch number: {}".format(epoc))
        for batch_idx, data in enumerate(dataloader):

            optimizer.zero_grad()
    
            channel_matrix_batch = data.matrix
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_links, num_links)

            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)   
            psi = psi.view(batch_size, num_links, output_dim)                  
          
            # Distribución sobre acciones
            probs = torch.softmax(psi, dim=-1)  

            # Muestreamos acciones
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()                           
            log_p = dist.log_prob(actions)                    
            log_p_sum = log_p.sum(dim=1).unsqueeze(-1)        

            phi = torch.zeros(batch_size, num_links, num_channels, device=probs.device)

            # Máscara de los que transmiten
            active_mask = (actions > 0)

            if active_mask.any():
                # índices de batch y de link de los activos
                batch_ids, link_ids = torch.nonzero(active_mask, as_tuple=True)

                # acción elegida para esos activos (shift -1 porque 0 era "no transmitir")
                actions_active = actions[batch_ids, link_ids] - 1  

                # separar en canal y potencia
                channel_idx = actions_active // num_power_levels
                power_idx   = actions_active %  num_power_levels

                # asignar potencia al canal correcto
                phi[batch_ids, link_ids, channel_idx] = power_levels.to(phi.device)[power_idx]

            # Calcula rates
            rates = nuevo_get_rates(phi, channel_matrix_batch, sigma)  

            # Objective
            sum_rate = objective_function(rates).unsqueeze(-1)  
            sum_rate_mean = torch.mean(sum_rate, dim=0)

            # Cálculo del costo y backprop - ahora solo considera sum_rate
            loss = sum_rate * log_p_sum                   
            loss_mean = torch.mean(loss, dim=0)

            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_idx % 10 == 0:
                probs_values.append(probs.mean(dim=[0,1]).detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())



    for name, param in gnn_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    path = plot_results(
            building_id=building_id,
            b5g=b5g,
            normalized_psi=probs,
            normalized_psi_values=probs_values,
            num_layers=num_layers,
            K=K,
            batch_size=batch_size,
            epochs=epochs,
            rn=rn,
            rn1=rn1,
            eps=eps,
            objective_function_values=objective_function_values,
            power_constraint_values=power_constraint_values,
            loss_values=loss_values,
            mu_k_values=mu_k_values,
            train=True
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
