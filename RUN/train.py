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
from utils import graphs_to_tensor_sc

def run(building_id=990, b5g=False, num_links=5, num_channels=3, num_layers=5, K=3,
        batch_size=64, epochs=100, eps=5e-5, mu_lr=5e-4, synthetic=1, rn=100, rn1=100,
         sigma=1e-4,max_antenna_power_dbm=6, sanitycheck=True):   

    banda = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str = str(f"{mu_lr:.0e}")

    if synthetic:
        if sanitycheck:
            x_tensor, channel_matrix_tensor = graphs_to_tensor_sc(
                num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
            )
            dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
            dataloader = DataLoader(dataset[:7000], batch_size=batch_size, shuffle=True, drop_last=True)

        else:
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



    max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
    pmax_per_ap = max_antenna_power_mw*0.8* torch.ones((num_links,))
    mu_k =  torch.ones((num_links,), requires_grad=False)  

    # ---- Definición de red ----
    input_dim = 1
    hidden_dim = 1


    # Eliminar niveles de potencia, solo selección categórica de canales
    output_dim = num_channels + 1  # [no_tx, canal_0, canal_1, canal_2, ...]
    
    dropout = True

    #------------------------------------------------------------------------

    # Versión 2
    # power_levels = torch.tensor([ max_antenna_power_mw/2, max_antenna_power_mw])  # Incluimos 0 = no transmitir
    # num_power_levels = len(power_levels)

    # output_dim = 2 + num_channels + num_power_levels


    #------------------------------------------------------------------------
    # En train.py:
    input_dim = num_links +1 # Era 1, ahora 3 (o el número de links)
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)

    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    probs_values = []
    power_values = []  


    for epoc in range(epochs):
        print("Epoch number: {}".format(epoc))
        for batch_idx, data in enumerate(dataloader):

            #------------------------------------------------------------------------

            # Versión 1
            channel_matrix_batch = data.matrix
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_links, num_links) 
                
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr) 
            psi = psi.view(batch_size, num_links, num_channels+1) 
            probs = torch.softmax(psi, dim=-1)  
            dist = torch.distributions.Categorical(probs=probs)  
            actions = dist.sample()            
            log_p = dist.log_prob(actions)     # [64, 5]
            log_p_sum = log_p.sum(dim=1).unsqueeze(-1)       # [64,1]
            
            # Solo selección de canal, sin niveles de potencia
            phi = torch.zeros(batch_size, num_links, num_channels, device=probs.device)
            active_mask = (actions > 0)
            active_channels = actions[active_mask] - 1
            phi[active_mask, active_channels] = max_antenna_power_mw  
            #------------------------------------------------------------------------

            #------------------------------------------------------------------------
            #Versión 2

            # channel_matrix_batch = data.matrix
            # channel_matrix_batch = channel_matrix_batch.view(batch_size, num_links, num_links) # [64, 5, 5]
            # psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)   # [batch*num_links, num_actions]
            # psi = psi.view(batch_size, num_links, output_dim)                  # [batch, num_links, num_actions]
          
            # psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            # psi = psi.view(batch_size, num_links, output_dim)

            # action_logits = psi                                                    # [batch, links, output_dim]
            # action_probs = torch.softmax(action_logits, dim=-1)                    # [batch, links, output_dim]
            # action_dist = torch.distributions.Categorical(probs=action_probs)
            # actions = action_dist.sample()                                         # [batch, links] - valores 0..(output_dim-1)
            # log_p_actions = action_dist.log_prob(actions)                          # [batch, links]

            # # Mascara de transmisión (acción != 0)
            # transmit_mask = (actions != 0)
            # # Log probability total por muestra (sumamos sobre links)
            # log_p_sum = log_p_actions.sum(dim=1).unsqueeze(-1)                     # [batch, 1]

            # # Construir phi (potencias por canal)
            # phi = torch.zeros(batch_size, num_links, num_channels)

            # # Solo asignar potencia si decide transmitir
            # transmit_indices = torch.where(transmit_mask)
            # if len(transmit_indices[0]) > 0:
            #     batch_idx_tx = transmit_indices[0]
            #     link_idx_tx = transmit_indices[1]
            #     selected_actions = actions[batch_idx_tx, link_idx_tx]             # valores 1..(C*P)
            #     # convertir a canal y potencia
            #     selected_minus1 = selected_actions - 1
            #     selected_channels = (selected_minus1 // num_power_levels).long()
            #     selected_powers_idx = (selected_minus1 % num_power_levels).long()
            #     # Obtener valores reales de potencia (ej. en mW)
            #     selected_powers_values = power_levels[selected_powers_idx].to(phi.dtype)
            #     # Asignar phi
            #     phi[batch_idx_tx, link_idx_tx, selected_channels] = selected_powers_values
            #------------------------------------------------------------------------



            power_constr_per_ap = power_constraint_per_ap(phi, pmax_per_ap)  # [batch_size, num_links]
            power_constr_per_ap_mean = torch.mean(power_constr_per_ap, dim=(0,1))  
            rates = nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=max_antenna_power_mw)

            sum_rate = -objective_function(rates).unsqueeze(-1) 
            sum_rate_mean = torch.mean(sum_rate, dim=0)

            # Actualización de la penalización
            mu_k = mu_update(mu_k, power_constr_per_ap, eps) 

            penalty_per_ap = power_constr_per_ap * mu_k.unsqueeze(0)  
            total_penalty = penalty_per_ap.sum(dim=1).unsqueeze(-1)   

            cost = sum_rate + total_penalty  
            
            loss = cost * log_p_sum        
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=5.0)  
            optimizer.step()

            # if batch_idx % 10 == 0:
            #     with torch.no_grad():
            #         # Ortogonal
            #         phi_ortho = torch.zeros_like(phi)
            #         phi_ortho[:, 0, 0] = max_antenna_power_mw
            #         phi_ortho[:, 1, 1] = max_antenna_power_mw
            #         phi_ortho[:, 2, 2] = max_antenna_power_mw
                    
            #         rates_ortho = nuevo_get_rates(phi_ortho, channel_matrix_batch, sigma, p0=max_antenna_power_mw)
            #         sum_rate_ortho = objective_function(rates_ortho).mean()  # SIN menos (rate crudo)
                    
            #         sum_rate_actual = objective_function(rates).mean()  # SIN menos (rate crudo)
                    
            #         print(f"Batch {batch_idx}: Rate ortho={sum_rate_ortho:.4f}, Rate actual={sum_rate_actual:.4f}, Diff={sum_rate_ortho - sum_rate_actual:.4f}")
            if batch_idx%10 == 0:

            #------------------------------------------------------------------------

                # Versión 1
                probs_mean_batch = probs.mean(dim=0)  # Promedio sobre batch: [num_links, num_channels+1]
                probs_values.append(probs_mean_batch.detach().numpy())
            #------------------------------------------------------------------------

            #------------------------------------------------------------------------

                # Versión 2
                # probs_mean_batch = action_probs.mean(dim=0)  # Promedio sobre batch: [num_links, num_channels+1]
                # probs_values.append(probs_mean_batch.detach().numpy())
                              
                # # Guardar también información de potencia (phi)
                # phi_mean_batch = phi.mean(dim=0)  # Promedio sobre batch: [num_links, num_channels]
                # power_values.append(phi_mean_batch.detach().numpy())
            #------------------------------------------------------------------------

                
                power_constraint_values.append(power_constr_per_ap_mean.detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().numpy())
    

    for name, param in gnn_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

# Versión 1
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
            mu_lr = mu_lr,
            objective_function_values=objective_function_values,
            power_constraint_values=power_constraint_values,
            loss_values=loss_values,
            mu_k_values=mu_k_values,
            train=True
        )


        

# Versión 2

    # path = plot_results(
    #         building_id=building_id,
    #         b5g=b5g,
    #         normalized_psi=psi, 
    #         normalized_psi_values=probs_values,
    #         power_values=power_values,  # ← AGREGA ESTA LÍNEA
    #         num_layers=num_layers,
    #         K=K,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         rn=rn,
    #         rn1=rn1,
    #         eps=eps,
    #         mu_lr=mu_lr,
    #         objective_function_values=objective_function_values,
    #         power_constraint_values=power_constraint_values,
    #         loss_values=loss_values,
    #         mu_k_values=mu_k_values,
    #         num_channels = num_channels,
    #         num_power_levels = num_power_levels,
    #         power_levels = power_levels
    #     )

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
    parser.add_argument('--num_links', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=5e-5)
    parser.add_argument('--mu_lr', type=float, default=5e-3)
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
    
    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, num_layers=args.num_layers, K=args.k, batch_size=args.batch_size, epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, rn=rn, rn1=rn1, sanitycheck=True)
    print('Seeds: {} and {}'.format(rn, rn1))
