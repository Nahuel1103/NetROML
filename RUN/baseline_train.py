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
from utils import nuevo_get_rates
from utils import graphs_to_tensor_synthetic

def run(building_id=990, b5g=False, num_links=5, num_channels=3, num_layers=5, K=3,
        batch_size=64, epochs=100, eps=5e-5, mu_lr=5e-4, baseline=1, synthetic=1, 
        rn=100, rn1=100, sigma=1e-4, max_antenna_power_dbm=6):   

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
    pmax_per_ap = max_antenna_power_mw * 0.6 * torch.ones((num_links,))
    mu_k = torch.ones((num_links,), requires_grad=False)  

    # Definir niveles de potencia discretos (igual que en train.py)
    power_levels = torch.tensor([0, max_antenna_power_mw/2, max_antenna_power_mw])
    num_power_levels = len(power_levels)

    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    tx_policy_values = []
    channel_policy_values = []
    power_policy_values = []

    for epoc in range(epochs):
        print("Epoch number: {}".format(epoc))
        for batch_idx, data in enumerate(dataloader):
            
            channel_matrix_batch = data.matrix
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_links, num_links)
            
            # BASELINE 1: Política uniforme simple
            if baseline == 1:
                # Probabilidades uniformes para todas las decisiones
                tx_probs = torch.ones(batch_size, num_links, 2) * 0.5  # 50% prob de transmitir
                channel_probs = torch.ones(batch_size, num_links, num_channels) / num_channels
                power_probs = torch.ones(batch_size, num_links, num_power_levels) / num_power_levels
                
            # BASELINE 2: Política greedy (siempre transmitir, canal con mejor SNR, máxima potencia)
            elif baseline == 2:
                tx_probs = torch.zeros(batch_size, num_links, 2)
                tx_probs[:, :, 1] = 1.0  # Siempre transmitir
                
                # Encontrar el canal con mejor SNR para cada enlace
                diagH = torch.diagonal(channel_matrix_batch, dim1=1, dim2=2)
                best_channels = torch.argmax(diagH, dim=-1)
                
                channel_probs = torch.zeros(batch_size, num_links, num_channels)
                for i in range(batch_size):
                    for j in range(num_links):
                        channel_probs[i, j, best_channels[i, j]] = 1.0
                
                # Siempre usar máxima potencia
                power_probs = torch.zeros(batch_size, num_links, num_power_levels)
                power_probs[:, :, -1] = 1.0  # Último nivel de potencia (máximo)

            # Muestrear decisiones (igual que en train.py)
            tx_dist = torch.distributions.Categorical(probs=tx_probs)
            channel_dist = torch.distributions.Categorical(probs=channel_probs)
            power_dist = torch.distributions.Categorical(probs=power_probs)

            tx_actions = tx_dist.sample()
            channel_actions = channel_dist.sample()
            power_actions = power_dist.sample()

            # Log probabilities (necesarias para el cálculo de pérdida)
            log_p_tx = tx_dist.log_prob(tx_actions)
            log_p_channel = channel_dist.log_prob(channel_actions)
            log_p_power = power_dist.log_prob(power_actions)

            transmit_mask = (tx_actions == 1)
            log_p_total = log_p_tx.clone()
            log_p_total[transmit_mask] += log_p_channel[transmit_mask] + log_p_power[transmit_mask]
            log_p_sum = log_p_total.sum(dim=1).unsqueeze(-1)

            # Construir phi (matriz de potencia por canal)
            phi = torch.zeros(batch_size, num_links, num_channels)
            transmit_indices = torch.where(transmit_mask)
            if len(transmit_indices[0]) > 0:
                batch_idx_tx = transmit_indices[0]
                link_idx_tx = transmit_indices[1]
                selected_channels = channel_actions[batch_idx_tx, link_idx_tx]
                selected_powers = power_actions[batch_idx_tx, link_idx_tx]
                phi[batch_idx_tx, link_idx_tx, selected_channels] = power_levels[selected_powers]

            # Calcular restricciones de potencia y tasas
            power_constr_per_ap = power_constraint_per_ap(phi, pmax_per_ap)
            power_constr_per_ap_mean = torch.mean(power_constr_per_ap, dim=(0,1))
            rates = nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=max_antenna_power_mw)

            sum_rate = -objective_function(rates).unsqueeze(-1)
            sum_rate_mean = torch.mean(sum_rate, dim=0)

            # Actualizar multiplicadores de Lagrange
            mu_k = mu_update(mu_k, power_constr_per_ap, eps)

            # Calcular coste y pérdida
            penalty_per_ap = power_constr_per_ap * mu_k.unsqueeze(0)
            total_penalty = penalty_per_ap.sum(dim=1).unsqueeze(-1)
            cost = sum_rate + total_penalty
            loss = cost * log_p_sum
            loss_mean = loss.mean()

            # Guardar métricas cada 10 batches
            if batch_idx % 10 == 0:
                tx_policy_values.append(tx_probs.mean(dim=[0,1]).detach().numpy())
                channel_policy_values.append(channel_probs.mean(dim=[0,1]).detach().numpy())
                power_policy_values.append(power_probs.mean(dim=[0,1]).detach().numpy())
                power_constraint_values.append(power_constr_per_ap_mean.detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().numpy())

    # Graficar resultados
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
        train=False,  # Indica que es baseline
        synthetic=synthetic,
        num_channels=num_channels,
        num_power_levels=num_power_levels,
        baseline=baseline  # Pasar información del baseline
    )

    # Guardar resultados
    file_name = path + 'baseline'+str(baseline)+'_'+str(epochs)+'.pkl'
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

    parser = argparse.ArgumentParser(description='Baseline configuration')

    parser.add_argument('--building_id', type=int, default=856)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=20)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--mu_lr', type=float, default=1e-3)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--baseline', type=int, default=1, choices=[1, 2],
                       help='1: Uniform policy, 2: Greedy policy')

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
    print(f'baseline: {args.baseline}')

    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, 
        num_channels=args.num_channels, num_layers=args.num_layers, K=args.k, 
        batch_size=args.batch_size, epochs=args.epochs, eps=args.eps, 
        mu_lr=args.mu_lr, baseline=args.baseline, synthetic=args.synthetic, 
        rn=rn, rn1=rn1)
    
    print('Seeds: {} and {}'.format(rn, rn1))