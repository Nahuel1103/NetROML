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

from plot_results_torch_v1 import plot_results
from gnn import GNN
from utils_v1 import graphs_to_tensor
from utils_v1 import get_gnn_inputs
from utils_v1 import power_constraint_per_ap
from utils_v1 import objective_function
from utils_v1 import mu_update
from utils_v1 import nuevo_get_rates


from utils_v1 import graphs_to_tensor_synthetic
from utils_v1 import graphs_to_tensor_sc

def run(building_id=990, b5g=False, num_links=5, num_channels=3, num_layers=5, K=3,
        batch_size=64, epochs=100, eps=5e-5, mu_lr=5e-4, synthetic=1, rn=100, rn1=100,
         sigma=1e-4,max_antenna_power_dbm=6, sanitycheck=True, num_noise_per_graph=10):   

    banda = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str = str(f"{mu_lr:.0e}")

    # Cargar datos
    if synthetic:
        if sanitycheck:
            x_tensor, channel_matrix_tensor = graphs_to_tensor_sc(
                num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
            )
        else:
            x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
                num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
            )
    else:
        x_tensor, channel_matrix_tensor = graphs_to_tensor(
            train=True, num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )

    # MODIFICADO: Ahora get_gnn_inputs genera múltiples samples de ruido por grafo
    dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor, 
                            num_noise_per_graph=num_noise_per_graph)
    
    # El batch_size efectivo se multiplica por num_noise_per_graph
    effective_batch_size = batch_size * num_noise_per_graph
    dataloader = DataLoader(dataset[:7000], batch_size=effective_batch_size, 
                           shuffle=True, drop_last=True)

    max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
    pmax_per_ap = max_antenna_power_mw*0.8* torch.ones((num_links,))
    mu_k =  torch.ones((num_links,), requires_grad=False)  

    # ---- Definición de red ----
    input_dim = 1
    hidden_dim = 1  # Puedes ajustarlo

    # Eliminar niveles de potencia, solo selección categórica de canales
    output_dim = num_channels + 1  # [no_tx, canal_0, canal_1, canal_2, ...]
    
    dropout = True
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

            # data.matrix contiene todas las matrices aplanadas concatenadas
            total_matrix_elements = data.matrix.shape[0]
            actual_num_links = int(np.sqrt(total_matrix_elements / effective_batch_size))
            
            if batch_idx == 0 and epoc == 0:
                print(f"INFO: num_links del argumento: {num_links}")
                print(f"INFO: num_links real en los datos: {actual_num_links}")
                print(f"INFO: Usando num_links = {actual_num_links} para el resto del entrenamiento")
            
            # Usar el num_links correcto
            num_links_actual = actual_num_links
            
            # Obtener channel matrices
            channel_matrix_all = data.matrix.view(effective_batch_size, num_links_actual, num_links_actual)
            
            # Tomar solo los grafos únicos (cada num_noise_per_graph es el mismo grafo)
            channel_matrix_batch = channel_matrix_all[::num_noise_per_graph]  # [batch_size, num_links_actual, num_links_actual]

            if batch_idx == 0 and epoc == 0:
                H_ejemplo = channel_matrix_batch[0]  # Primer grafo del primer batch
                eigenvalues, eigenvectors = torch.linalg.eigh(H_ejemplo)
                print("=== Vectores propios del primer grafo ===")
                print("Eigenvalues:", eigenvalues)
                print("Eigenvectors (cada columna es un u_n):\n", eigenvectors)
                print("Shape:", eigenvectors.shape)

            # 1. Forward con todos los ruidos
            # CLAVE: La salida de GNN en PyG es [total_nodes, output_dim]
            # donde total_nodes = effective_batch_size * num_links_actual
            psi_raw = gnn_model.forward(data.x, data.edge_index, data.edge_attr) 
            # psi_raw: [effective_batch_size * num_links_actual, output_dim]
            
            # 2. Reshape para [effective_batch_size, num_links_actual, output_dim]
            psi_raw = psi_raw.view(effective_batch_size, num_links_actual, output_dim)
            
            # 3. Reshape para separar los samples de ruido
            psi_raw = psi_raw.view(batch_size, num_noise_per_graph, num_links_actual, output_dim)
            
            # 4. Aplicar σ(·) = (·)² coordenada a coordenada
            psi_squared = psi_raw ** 2
            
            # 5. Promediar sobre los samples de ruido: E[σ(z)]
            psi = psi_squared.mean(dim=1)  # [batch_size, num_links_actual, output_dim]
            # Esta es la clave del paper: E[z²] mantiene equivarianza
            
            probs = torch.softmax(psi, dim=-1)  
            dist = torch.distributions.Categorical(probs=probs)  
            actions = dist.sample()            
            log_p = dist.log_prob(actions)     # [batch_size, num_links_actual]
            log_p_sum = log_p.sum(dim=1).unsqueeze(-1)       # [batch_size, 1]
            
            # Solo selección de canal, sin niveles de potencia
            phi = torch.zeros(batch_size, num_links_actual, num_channels, device=probs.device)
            active_mask = (actions > 0)
            active_channels = actions[active_mask] - 1
            
            # Crear índices para la asignación
            batch_indices = torch.arange(batch_size, device=probs.device).unsqueeze(1).expand(-1, num_links_actual)[active_mask]
            link_indices = torch.arange(num_links_actual, device=probs.device).unsqueeze(0).expand(batch_size, -1)[active_mask]
            
            phi[batch_indices, link_indices, active_channels] = max_antenna_power_mw  


            power_constr_per_ap = power_constraint_per_ap(phi, pmax_per_ap[:num_links_actual])  # [batch_size, num_links_actual]
            power_constr_per_ap_mean = torch.mean(power_constr_per_ap, dim=(0,1))  
            rates = nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=max_antenna_power_mw)

            sum_rate = -objective_function(rates).unsqueeze(-1) 
            sum_rate_mean = torch.mean(sum_rate, dim=0)

            # Actualización de la penalización
            mu_k_actual = mu_k[:num_links_actual]
            mu_k_actual = mu_update(mu_k_actual, power_constr_per_ap, eps) 
            mu_k[:num_links_actual] = mu_k_actual

            penalty_per_ap = power_constr_per_ap * mu_k_actual.unsqueeze(0)  
            total_penalty = penalty_per_ap.sum(dim=1).unsqueeze(-1)   

            cost = sum_rate + total_penalty  
            
            loss = cost * log_p_sum        
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=5.0)  
            optimizer.step()

            if batch_idx%10 == 0:

                probs_mean_batch = probs.mean(dim=0)  # Promedio sobre batch: [num_links_actual, num_channels+1]
                probs_values.append(probs_mean_batch.detach().numpy())
                
                power_constraint_values.append(power_constr_per_ap_mean.detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k_actual.squeeze(-1).detach().numpy())
    
    print("\n=== Parámetros aprendidos ===")
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
            mu_lr = mu_lr,
            objective_function_values=objective_function_values,
            power_constraint_values=power_constraint_values,
            loss_values=loss_values,
            mu_k_values=mu_k_values,
            train=True
        )


    # file_name = path + 'objective_function_values_train_' + str(epochs) + '.pkl'
    # with open(file_name, 'wb') as archivo:
    #     pickle.dump(objective_function_values, archivo)

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
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=5e-4)
    parser.add_argument('--mu_lr', type=float, default=5e-3)
    parser.add_argument('--synthetic', type=int, default=1)
    parser.add_argument('--num_noise_per_graph', type=int, default=10,
                       help='Número de samples de ruido blanco por grafo (estrategia espectral)')
    
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
    print(f'num_noise_per_graph: {args.num_noise_per_graph}')
    
    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, 
        num_layers=args.num_layers, K=args.k, batch_size=args.batch_size, 
        epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, 
        rn=rn, rn1=rn1, sanitycheck=True, num_noise_per_graph=args.num_noise_per_graph)
    
    print('Seeds: {} and {}'.format(rn, rn1))