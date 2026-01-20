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
from MultiHeadGNN import MultiHeadGNN
from utils import (graphs_to_tensor, get_gnn_inputs, objective_function, 
                   power_constraint_per_ap, nuevo_get_rates, mu_update_per_ap, 
                   graphs_to_tensor_synthetic)
from plot_results_torch import plot_results


def run(building_id=990, b5g=False, num_links=5, num_channels=3, num_layers=5, K=3,
        batch_size=64, epochs=100, eps=5e-5, mu_lr=1e-4, synthetic=1, 
         sigma=1e-4,max_antenna_power_dbm=6, num_power_levels=2):   

    banda = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str = str(f"{mu_lr:.0e}")

    # --- 1. Configuración de Dispositivo ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    ############# DataLoader #############
    if synthetic:
        x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
            num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataset = dataset[:7000]
    else:
        x_tensor, channel_matrix_tensor = graphs_to_tensor(
            train=True, num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    ############# Parámetros de optimización #############
    mu_k = torch.ones((num_links,), requires_grad=False).to(device)  # [num_links]

    #---- Parámetros del entorno ----
    p0 = 10**(max_antenna_power_dbm/10)
    pmax_per_ap = (0.8 * p0 * torch.ones((num_links,))).to(device)  # [num_links]

    # Niveles discretos de potencia
    power_levels = torch.linspace(
            p0/num_power_levels,
            p0,
            num_power_levels,
            dtype=torch.float32
        ).to(device)  # [num_power_levels]

    # ---- Definición de red ----
    input_dim = 1
    hidden_dim = 64

    dropout = False
    
    gnn_model = MultiHeadGNN(input_dim, hidden_dim, num_channels, num_power_levels, num_layers, dropout, K).to(device)
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)


    for name, param in gnn_model.named_parameters():
        param.requires_grad = True
        if param.requires_grad:
            print(name, param.data)


    # Listas para métricas
    metrics = {
        'objective': [], 'power_constraint': [], 'loss': [], 'mu_k': [], 'probs_ch': [], 'probs_pow': []
    }  

  #  ------------- Training -----------------
    for epoch in range(epochs):

        print("Epoch number: {}".format(epoch))
        for batch_idx, data in enumerate(dataloader):
    
            channel_matrix_batch = data.matrix
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_links, num_links) #[batch, num_links, num_links]

            # Forward Pass - Retorna dos juegos de Logits
            channel_logits, power_logits = gnn_model(data.x, data.edge_index, data.edge_attr)

            # [batch_size, num_links, output_dim]
            channel_logits = channel_logits.view(batch_size, num_links, num_channels + 1) # [batch, num_links, num_channels + 1]]
            power_logits = power_logits.view(batch_size, num_links, num_power_levels) # [batch, num_links, num_power_levels]

            # --- Muestreo Estocástico (Policy) ---
    
            # 1. Muestreo de Canal (incluye No_Tx)
            probs_ch = F.softmax(channel_logits, dim=-1)
            probs_ch = torch.clamp(probs_ch, min=1e-8)
            dist_ch = torch.distributions.Categorical(probs=probs_ch)
            actions_ch = dist_ch.sample() # [batch, num_links]
            log_p_ch = dist_ch.log_prob(actions_ch) # [batch, num_links]

            # 2. Muestreo de Potencia
            probs_pow = F.softmax(power_logits, dim=-1)
            probs_pow = torch.clamp(probs_pow, min=1e-8)
            dist_pow = torch.distributions.Categorical(probs=probs_pow)
            actions_pow = dist_pow.sample() # [batch, num_links]
            log_p_pow = dist_pow.log_prob(actions_pow) # [batch, num_links]

            # --- Decodificación de Acciones ---

            # 1. Máscara de Transmisión (si el canal muestreado NO es 0/No_Tx)
            active_mask = (actions_ch > 0)

            phi = torch.zeros((batch_size, num_links, num_channels), device=data.x.device) # [batch, num_links, num_channels]

            if active_mask.any():
                # Índice del canal activo: restamos 1 porque la acción 0 es No_Tx
                channel_idx = (actions_ch[active_mask] - 1).long()

                # Índice del nivel de potencia activo
                power_idx = actions_pow[active_mask].long()

                # Asignación de Potencia: phi[batch_idx, link_idx, channel_idx] = power_levels[power_idx]
                power_values_to_assign = power_levels[power_idx].to(phi.device)
                
                # Creo un tensor auxiliar para mapear los índices
                batch_indices = torch.where(active_mask)[0]
                link_indices = torch.where(active_mask)[1]
                
                # Asignar los valores de potencia
                phi[batch_indices, link_indices, channel_idx] = power_values_to_assign
            
            # --- Cálculo de Loss (REINFORCE) ---
    
            # Si se transmite, el log_p_total es la suma de log_p_ch y log_p_pow.
            # Si NO se transmite (actions_ch == 0), solo se cuenta log_p_ch.
            
            log_p_total = log_p_ch + active_mask * log_p_pow
            log_p_sum = log_p_total.sum(dim=1).unsqueeze(-1)
            
            power_constr_per_ap = power_constraint_per_ap(phi, pmax_per_ap)  # [batch, num_links]

            rates = nuevo_get_rates(phi, channel_matrix_batch, p0=p0) # [batch, num_links]

            objetive_rate = -objective_function(rates).unsqueeze(-1)  # [batch, 1]

            penalty_per_ap = power_constr_per_ap * mu_k.unsqueeze(0)  # [batch_size, num_links]
            total_penalty = penalty_per_ap.sum(dim=1).unsqueeze(-1)   # [batch_size, 1]

            # Costo total calculado (Lagrangiano)
            cost = objetive_rate + total_penalty # [batch_size, 1]

            # # Cálculo de la línea base del batch
            # b = cost.mean().detach() # Escalar, detached

            # # Pérdida REINFORCE centrada
            # loss = (cost - b) * log_p_sum 
            loss = cost * log_p_sum

            # Usamos la media para obtener un escalar de pérdida
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            mu_k = mu_update_per_ap(mu_k, power_constr_per_ap.detach(), eps) # [num_links]
            optimizer.step()

            if batch_idx % 10 == 0:
                metrics['objective'].append(-objetive_rate.detach().mean().item())
                metrics['power_constraint'].append(power_constr_per_ap.detach().mean().item())
                metrics['loss'].append(loss_mean.detach().item())
                metrics['mu_k'].append(mu_k.detach().mean().item())
                metrics['probs_ch'].append(probs_ch.detach().mean().item())
                metrics['probs_pow'].append(probs_pow.detach().mean().item())


    for name, param in gnn_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # path = plot_results(
    #         building_id=building_id,
    #         b5g=b5g,
    #         normalized_psi=probs,
    #         normalized_psi_values=probs_values,
    #         num_layers=num_layers,
    #         K=K,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         rn=rn,
    #         rn1=rn1,
    #         eps=eps,
    #         objective_function_values=objective_function_values,
    #         power_constraint_values=power_constraint_per_ap_values,
    #         loss_values=loss_values,
    #         mu_k_values=mu_k_values,
    #         train=True
    #     )

    # file_name = path + 'objective_function_values_train_' + str(epochs) + '.pkl'
    # with open(file_name, 'wb') as archivo:
    #     pickle.dump(metrics['objective'], archivo)

    # # save trained gnn weights in .pth
    # torch.save(gnn_model.state_dict(), path + 'gnn_model_weights.pth')

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
    parser.add_argument('--b5g', type=bool, default=False)
    parser.add_argument('--num_links', type=int, default=5)
    parser.add_argument('--num_channels', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--eps', type=float, default=5e-4)
    parser.add_argument('--mu_lr', type=float, default=5e-4)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=1e-4)
    parser.add_argument('--max_antenna_power_dbm', type=float, default=6)
    parser.add_argument('--num_power_levels', type=int, default=2)

    args = parser.parse_args()

    print(f'building_id: {args.building_id}')
    print(f'b5g: {args.b5g}')
    print(f'num_links: {args.num_links}')
    print(f'num_channels: {args.num_channels}')
    print(f'num_layers: {args.num_layers}')
    print(f'K: {args.K}')
    print(f'batch_size: {args.batch_size}')
    print(f'epochs: {args.epochs}')
    print(f'eps: {args.eps}')
    print(f'mu_lr: {args.mu_lr}')
    print(f'synthetic: {args.synthetic}')
    print(f'sigma: {args.sigma}')
    print(f'max_antenna_power_dbm: {args.max_antenna_power_dbm}')
    print(f'num_power_levels: {args.num_power_levels}') 


    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, num_channels=args.num_channels, num_layers=args.num_layers, K=args.K, batch_size=args.batch_size, epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, sigma=args.sigma, max_antenna_power_dbm=args.max_antenna_power_dbm, num_power_levels=args.num_power_levels)
    
    print('Seeds: {} and {}'.format(rn, rn1))