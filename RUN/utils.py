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

def transform_matrix(adj_matrix, all = True):

    if all:
        # primero defino quien es la pareja de quien.
        # recorro fila por fila la matriz y guardo en una lista el valor con mayor h y su posicion en la fila
        nodos = adj_matrix.shape[0]
        lista_parejas = []
        for i in range(nodos):
            # guardo el indice del nodo con menor h con el nodo i
            receptor = np.argmax(adj_matrix[i,:])
            # guardo el valor
            valor_maximo = adj_matrix[i,receptor]
            transmisor = i
            # guardo en una tupla el valor de h, el indice del transmisor y el indice del receptor
            elemento = [valor_maximo, transmisor, receptor]
            lista_parejas.append(elemento)

        H = np.zeros_like(adj_matrix)

        # voy rellenando la matriz
        for i in range(nodos):
            for j in range(nodos):
                if (i == j):
                    # en la diagonal uso los valores de h de lista_parejas con transmisor igual a i
                    H[i,j] = lista_parejas[i][0]
                else:
                    # en la no diagonal busco quien es el receptor del nodo j y relleno la matriz H con el h entre el nodo i y el receptor hallado
                    k = 0
                    nodo_receptor = 0
                    for k in range(nodos):
                        # busco la pareja que cumple que el nodo j es transmisor
                        if (lista_parejas[k][1] == j):
                            # guardo el nodo que recibe de j
                            nodo_receptor = lista_parejas[k][2]
                            if (nodo_receptor == i):
                                H[i,j] = 0.005
                            else:
                                H[i,j] = adj_matrix[i,nodo_receptor]
        return H
    
    else:
        # primero defino quien es transmisor y quien es receptor
        # defino los nodos
        num_nodes = adj_matrix.shape[0]
        nodos = list(np.arange(num_nodes))
        # barajo la lista
        random.seed(42)
        random.shuffle(nodos)
        # divido la lista barajada en dos listas
        half_nodes= num_nodes // 2
        nodos_tx = nodos[:half_nodes]
        nodos_rx = nodos[half_nodes:]

        # primero defino quien es la pareja de quien.
        # recorro fila por fila la matriz y guardo en una lista el valor con mayor h y su posicion en la fila
        #nodos = adj_matrix.shape[0]
        lista_parejas = []
        for nodo_tx in nodos_tx:
            h_canal = []
            for nodo_rx in nodos_rx:
                h_canal.append(adj_matrix[nodo_tx,nodo_rx])
            index_pareja = np.argmax(h_canal)
            valor_maximo = adj_matrix[nodo_tx, nodos_rx[index_pareja]]
            # guardo en una tupla el valor de h, el indice del transmisor y el indice del receptor
            elemento = [valor_maximo, nodo_tx, nodos_rx[index_pareja]]
            lista_parejas.append(elemento)

        # defino matriz H que se usa en el articulo
        H = np.zeros((num_nodes,num_nodes))

        for i in np.arange(half_nodes):
            nodo_tx = lista_parejas[i][1]
            for j in np.arange(half_nodes):
                if (i == j):
                    H[i,j] = lista_parejas[i][0]
                else:
                    # si no estas en la diagonal, necesito el h entre el transmisor i y el receptor del transmisor j
                    nodo_tx_a_j = nodos_tx[j]
                    receptor_j = -1
                    # busco el nodo receptor del nodo transmisor j
                    for k in np.arange(half_nodes):
                        if lista_parejas[k][1] == nodo_tx_a_j:
                            receptor_j = k
                    H[i,j] = adj_matrix[nodos_tx[i], nodos_rx[receptor_j]]
        return H
    
def graphs_to_tensor(train=True, num_channels=5, num_features=1, b5g=False, building_id=990):
    
    band = ['2_4', '5']
    path = '/Users/nahuelpineyro/NetROML/graphs/' + str(band[b5g]) + '_' + str(building_id) + '/'

    if (train):
        file_name = 'train_' + str(band[b5g]) + '_graphs_' + str(building_id) + '.pkl'
        #file_name = str(band[b5g]) + '_graphs_' + str(building_id) + '.pkl'
        with open(path + file_name, 'rb') as archivo:
            graphs = pickle.load(archivo)
    else:
        file_name = 'val_' + str(band[b5g]) + '_graphs_' + str(building_id) + '.pkl'
        with open(path + file_name, 'rb') as archivo:
            graphs = pickle.load(archivo)
    x_list = []
    channel_matrix_list = []
    x = torch.zeros((num_channels,num_features))
    for graph in graphs:
        adj_matrix = nx.adjacency_matrix(graph, weight = 'Atenuacion')
        adj_matrix = adj_matrix.toarray()
        channel_matrix = transform_matrix(adj_matrix, all = True)
        channel_matrix = channel_matrix/1e-3
        channel_matrix_list.append(torch.tensor(channel_matrix.T))
        x_list.append(x)

    channel_matrix_tensor = torch.stack(channel_matrix_list)
    x_tensor = torch.stack(x_list)
    return x_tensor, channel_matrix_tensor


def graphs_to_tensor_synthetic(num_channels, num_features = 1, b5g = False, building_id = 990):
    
    band = ['2_4', '5']
    path = '/Users/nahuelpineyro/NetROML/graphs/' + str(band[b5g]) + '_' + str(building_id) + '/'
    file_name = 'synthetic_graphs.pkl'
    with open(path + file_name, 'rb') as archivo:
        graphs = pickle.load(archivo)
    x_list = []
    channel_matrix_list = []
    x = torch.zeros((num_channels,num_features)) 
    for graph in graphs:
        channel_matrix_list.append(torch.tensor(graph))
        x_list.append(x)
    channel_matrix_tensor = torch.stack(channel_matrix_list)
    x_tensor = torch.stack(x_list)
    return x_tensor, channel_matrix_tensor

def get_gnn_inputs(x_tensor, channel_matrix_tensor):
    input_list = []
    size = channel_matrix_tensor.shape[0]
    for i in range(size):
        x = x_tensor[i,:,:]
        channel_matrix = channel_matrix_tensor[i,:,:]
        norm = np.linalg.norm(channel_matrix, ord = 2, axis = (0,1))
        channel_matrix_norm = channel_matrix / norm
        channel_matrix_norm = channel_matrix
        edge_index = channel_matrix_norm.nonzero(as_tuple=False).t()
        edge_attr = channel_matrix_norm[edge_index[0], edge_index[1]]
        edge_attr = edge_attr.to(torch.float)
        input_list.append(Data(matrix=channel_matrix, x=x, edge_index=edge_index, edge_attr=edge_attr))
    return input_list

def objective_function(rates):
    sum_rate = -torch.sum(rates, dim=1)
    return sum_rate

# def power_constraint(phi, pmax):
#     sum_phi = torch.sum(phi, dim=1)
#     return (sum_phi - pmax)

def power_constraint(phi, pmax):
    sum_phi = torch.sum(phi, dim=(1,2))
    return (sum_phi - pmax)

def mu_update(mu_k, power_constr, eps):
    mu_k = mu_k.detach()
    mu_k_update = eps * torch.mean(power_constr, dim = 0)
    mu_k = mu_k + mu_k_update
    mu_k = torch.max(mu_k, torch.tensor(0.0))
    return mu_k

def get_rates(phi, channel_matrix_batch, sigma):
    phi = torch.squeeze(phi, dim = 2)
    numerator = torch.unsqueeze(torch.diagonal(channel_matrix_batch, dim1=1, dim2=2) * phi, dim=2)
    expanded_phi = torch.unsqueeze(phi, dim=2)
    denominator = torch.matmul(channel_matrix_batch.float(), expanded_phi.float()) - numerator + sigma
    rates = torch.log(numerator / denominator + 1)
    return rates

# def nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=4):
#     # phi = torch.squeeze(phi, dim = 2) #(64,5,4) , channel_matrix_batch (64,5,5)
#     numerator = torch.unsqueeze(torch.diagonal(channel_matrix_batch, dim1=1, dim2=2) * phi, dim=2)
#     expanded_phi = phi.unsqueeze(2)
#     denominator = (torch.matmul(channel_matrix_batch.float(), expanded_phi.float()) - numerator)*expanded_phi/p0 + sigma
#     rates = torch.log(numerator / denominator + 1)
#     return rates

# Lo que había hecho yo:
# def nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=4):
#     # phi: [batch_size, num_links, output_dim]
#     # channel_matrix_batch: [batch_size, num_links, num_links]
#     channel_matrix_batch = channel_matrix_batch.float()
#     phi = phi.float()
    
#     batch_size, num_links, output_dim = phi.shape

#     rates = torch.zeros((batch_size, num_links), device=phi.device)

#     for k in range(output_dim):  # Para cada canal
#         # phi_k: [batch_size, num_links] (nodos que usan canal k)
#         phi_k = phi[:, :, k]  # Potencia asignada al canal k para cada nodo

#         # h_ii: [batch_size, num_links] (ganancia directa en canal k)
#         h_ii = torch.diagonal(channel_matrix_batch, dim1=1, dim2=2)

#         # Interferencia: suma de potencias de otros nodos en canal k
#         # channel_matrix_batch: [batch_size, num_links, num_links]
#         # phi_k: [batch_size, num_links]
#         interference = torch.matmul(channel_matrix_batch, phi_k.unsqueeze(2)).squeeze(2) - h_ii * phi_k

#         # Numerador y denominador según la fórmula
#         numerator = h_ii * phi_k
#         denominator = sigma + interference * (phi_k / p0)

#         # Solo sumar la tasa si el nodo está usando ese canal (phi_k > 0)
#         rates += torch.log1p(numerator / (denominator + 1e-10)) * (phi_k > 0).float()

#     return rates  # [batch_size, num_links]

# Lo que hizo mauri:
# def nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=4):
#     """
#     phi: [batch_size, num_links, num_channels] (potencia por canal)
#     channel_matrix_batch: [batch_size, num_links, num_links] (ganancias |h_ji|^2)
#     sigma: ruido
#     p0: potencia por canal cuando está activo
#     """
#     batch_size, num_links, num_channels = phi.shape
    
#     # Calcular potencia total por enlace (p_i)
#     p_i = torch.sum(phi, dim=2)  # [64, 5]
#     # Obtener ganancias directas (|h_ii|^2)
#     diag_gains = torch.diagonal(channel_matrix_batch, dim1=1, dim2=2)  # [64, 5]
    
#     # Calcular numerador |h_ii|^2 * p_i
#     numerator = diag_gains * p_i  # [64, 5]
#     # Calcular interferencia (término ∑)
#     interference = torch.zeros(batch_size, num_links, device=phi.device)
    
#     for ch in range(num_channels):
#         # Máscara para enlaces transmitiendo en este canal
#         transmitting = (phi[:, :, ch] > 0).float()  # [batch_size, num_links]
        
#         # Potencia transmitida en este canal (p0 o 0)
#         p_ch = transmitting*p0  # [batch_size, num_links]
        
#         # Calcular interferencia generada por este canal
#         interf_ch = torch.matmul(channel_matrix_batch.float(), p_ch.unsqueeze(2).float()).squeeze(2)  # [batch_size, num_links]
        
#         # Restar auto-interferencia
#         interf_ch = torch.abs(interf_ch - diag_gains * p_ch)
        
#         # Aplicar factor de escala (p_i^T/p0)
#         scale_factor = (phi[:, :, ch] > 0).float()  # [batch_size, num_links]
#         interf_ch = interf_ch * (p_i / p0) * scale_factor
#         interference += interf_ch
#     # Calcular tasa final
#     rates = torch.log1p(numerator / (sigma + interference))  # [batch_size, num_links]
#     return rates


# Forma vectorizada de nuevo_get_rates
def nuevo_get_rates(phi: torch.Tensor,
                           H: torch.Tensor,
                           sigma: float,
                           p0: float = 4.0) -> torch.Tensor:
    """
    Versión vectorizada de get_rates.

    Args:
        phi: Tensor[batch, m, c], asignación de potencia (0 o p0) por enlace y canal.
        H:   Tensor[batch, m, m], matriz de ganancias |h_ji|^2.
        sigma: ruido térmico (escalar).
        p0:  potencia por canal cuando está activo.
        eps: término pequeño para evitar div/0.

    Returns:
        rates: Tensor[batch, m], tasa de cada enlace.
    """
    H   = H.float()
    phi = phi.float()
    
    batch, m, c = phi.shape

    # 1) Potencia total por enlace: p_i = Σ_k φ[i,k]
    p_i = phi.sum(dim=2)                                   # [batch, m]

    # 2) Diagonal de H: ganancias directas |h_ii|^2
    diag_gains = torch.diagonal(H, dim1=1, dim2=2)         # [batch, m]

    # 3) Numerador: |h_ii|^2 * p_i
    numerator = diag_gains * p_i                           # [batch, m]

    # 4) Interferencia por canal: 
    #    batch-matmul H [batch, m, m] x phi [batch, m, c] → [batch, m, c]
    interf_all = torch.matmul(H, phi)                      # [batch, m, c]

    # 5) Restar auto‐interferencia channel‐by‐channel:
    #    interf_ch = | interf_all - diag_gains[...,None] * phi |
    interf_self_sub = torch.abs(interf_all - diag_gains.unsqueeze(-1) * phi)

    # 6) Escalado por (p_i / p0) solamente donde φ>0:
    mask   = (phi > 0).float()                             # [batch, m, c]
    scale  = (p_i.unsqueeze(-1) / (p0)) * mask       # [batch, m, c]
    interference = (interf_self_sub * scale).sum(dim=2)    # [batch, m]

    # 7) Tasa final: log(1 + SINR)
    sinr  = numerator / (sigma + interference)       # [batch, m]
    rates = torch.log1p(sinr)                              # [batch, m]

    return rates
