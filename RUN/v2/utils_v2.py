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
import matplotlib.pyplot as plt
import os

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
    
def graphs_to_tensor(train=True, num_links=5, num_features=1, b5g=False, building_id=990):
    
    band = ['2_4', '5']
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    GRAPH_ROOT = os.path.join(BASE_DIR, '../..', 'graphs')
    path = os.path.join(GRAPH_ROOT, f'{band[b5g]}_{building_id}')

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
    x = torch.zeros((num_links,num_features))
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


def graphs_to_tensor_synthetic(num_links, num_features = 1, b5g = False, building_id = 990):
    
    band = ['2_4', '5']
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    GRAPH_ROOT = os.path.join(BASE_DIR, '../..', 'graphs')
    path = os.path.join(GRAPH_ROOT, f'{band[b5g]}_{building_id}')
    file_name = 'synthetic_graphs.pkl'
    with open(os.path.join(path, file_name), 'rb') as archivo:
        graphs = pickle.load(archivo)
    x_list = []
    channel_matrix_list = []
    x = torch.zeros((num_links,num_features)) 
    for graph in graphs:
        channel_matrix_list.append(torch.tensor(graph))
        x_list.append(x)
    channel_matrix_tensor = torch.stack(channel_matrix_list)
    x_tensor = torch.stack(x_list)
    return x_tensor, channel_matrix_tensor


def graphs_to_tensor_sc(num_links, num_features = 1, b5g = False, building_id = 990):
    
    band = ['2_4', '5']
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    GRAPH_ROOT = os.path.join(BASE_DIR, '../..', 'graphs')
    path = os.path.join(GRAPH_ROOT, f'{band[b5g]}_{building_id}')
    file_name = 'sc_graphs.pkl'
    with open(os.path.join(path, file_name), 'rb') as archivo:
        graphs = pickle.load(archivo)
    x_list = []
    channel_matrix_list = []
    x = torch.zeros((num_links,num_features)) 
    for graph in graphs:
        channel_matrix_list.append(torch.tensor(graph))
        x_list.append(x)
    channel_matrix_tensor = torch.stack(channel_matrix_list)
    x_tensor = torch.stack(x_list)
    return x_tensor, channel_matrix_tensor




# def get_gnn_inputs(x_tensor, channel_matrix_tensor):
#     input_list = []
#     size = channel_matrix_tensor.shape[0]
#     for i in range(size):
#         x = x_tensor[i,:,:]
#         channel_matrix = channel_matrix_tensor[i,:,:]
#         norm = np.linalg.norm(channel_matrix, ord = 2, axis = (0,1))
#         channel_matrix_norm = channel_matrix / norm
#         channel_matrix_norm = channel_matrix
#         edge_index = channel_matrix_norm.nonzero(as_tuple=False).t()
#         edge_attr = channel_matrix_norm[edge_index[0], edge_index[1]]
#         edge_attr = edge_attr.to(torch.float)
#         input_list.append(Data(matrix=channel_matrix, x=x, edge_index=edge_index, edge_attr=edge_attr))
#     return input_list


# def get_gnn_inputs(x_tensor, channel_matrix_tensor):
#     input_list = []
#     size = channel_matrix_tensor.shape[0]
#     num_links = x_tensor.shape[1]
    
#     for i in range(size):
#         channel_matrix = channel_matrix_tensor[i,:,:]
        
#         # Features: [one-hot ID, canal directo]
#         node_id = torch.eye(num_links)  # [3, 3]
#         diag = torch.diagonal(channel_matrix).unsqueeze(1)  # [3, 1]
#         x = torch.cat([node_id, diag], dim=1).float()  
        
#         norm = np.linalg.norm(channel_matrix, ord = 2, axis = (0,1))
#         channel_matrix_norm = channel_matrix / norm
#         edge_index = channel_matrix_norm.nonzero(as_tuple=False).t()
#         edge_attr = channel_matrix_norm[edge_index[0], edge_index[1]].float()  # Y ACÁ
        
#         input_list.append(Data(matrix=channel_matrix, x=x, edge_index=edge_index, edge_attr=edge_attr))
#     return input_list




def get_gnn_inputs(x_tensor, channel_matrix_tensor, K=3):
    """
    Features: k-hop closed loops
    Equivariante y captura estructura local/global
    """
    input_list = []
    size = channel_matrix_tensor.shape[0]
    num_links = x_tensor.shape[1]
    
    for i in range(size):
        channel_matrix = channel_matrix_tensor[i, :, :]
        
        # ✅ EQUIVARIANTE: k-hop loops
        features_list = []
        for k in range(2*K - 1):  # k = 0, 1, 2, 3, 4 (si K=3)
            if k == 0:
                # S^0 = I → diagonal = [1,1,1,...]
                loop_k = torch.ones(num_links, 1)
            else:
                # S^k: loops cerrados de k-hops
                S_k = torch.linalg.matrix_power(channel_matrix.float(), k)
                loop_k = torch.diagonal(S_k).unsqueeze(1)
            
            features_list.append(loop_k)
        
        x = torch.cat(features_list, dim=1).float()  # [6, 5]
        
        # x[:, 0] = 1          (constante)
        # x[:, 1] = h_ii       (ganancia directa)
        # x[:, 2] = Σh_ij*h_ji (2-hop loops)
        # x[:, 3] = ...        (3-hop loops)
        # x[:, 4] = ...        (4-hop loops)
        
        # Normalizar
        norm = torch.linalg.matrix_norm(channel_matrix, ord=2)
        channel_matrix_norm = channel_matrix / (norm + 1e-12)
        
        # Construir grafo
        edge_index = channel_matrix_norm.nonzero(as_tuple=False).t()
        edge_attr = channel_matrix_norm[edge_index[0], edge_index[1]].float()
        
        input_list.append(Data(
            matrix=channel_matrix,
            x=x,                    
            edge_index=edge_index,
            edge_attr=edge_attr
        ))


    return input_list





def objective_function(rates):
    # sumamos solo sobre links
    sum_rate = torch.sum(rates, dim=1)  # [batch_size]
    return sum_rate


def power_constraint_per_ap(phi, pmax_per_ap):
    sum_phi_per_link = torch.sum(phi, dim=2)  
    return (sum_phi_per_link - pmax_per_ap)

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




# Versión 3
def nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=4, alpha=0.3, p_rx_threshold=1e-1, eps=1e-12):
    """Versión corregida de nuevo_get_rates"""
    batch_size, num_links, num_channels = phi.shape

    # Señal útil
    diagH = torch.diagonal(channel_matrix_batch, dim1=1, dim2=2)
    signal = diagH.unsqueeze(-1) * phi

    # Interferencia intra-canal
    interf_same = torch.einsum('bij,bjc->bic', channel_matrix_batch.float(), phi.float()) - signal

    # Interferencia por canales solapados
    interf_overlap = torch.zeros_like(interf_same)
    if num_channels > 1:
        left_shift = torch.roll(phi, shifts=1, dims=2)
        right_shift = torch.roll(phi, shifts=-1, dims=2)
        interf_overlap = alpha * (left_shift + right_shift)
    interf_overlap *= alpha

    denom = sigma + interf_same + interf_overlap
    
    snr = signal / (denom + eps)

    for c in range(num_channels):
        p_ch = phi[:, :, c]
        recv_power = channel_matrix_batch * p_ch.unsqueeze(1)
        
        tx_active = p_ch > eps
        seen = (recv_power >= p_rx_threshold) & (tx_active.unsqueeze(1))
        seen_count = seen.sum(dim=-1) 
        
        invalid = seen_count >= 2
        
        snr[:, :, c] = snr[:, :, c] * (~invalid).float()  

    # Tasa por enlace
    rates = torch.log1p(torch.sum(snr, dim=-1))  # [batch, links]
    return rates


