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
    path = '/Users/mauriciovieirarodriguez/project/NetROML/Primer problema de optimización/graphs/' + str(band[b5g]) + '_' + str(building_id) + '/'

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
    path = '/Users/mauriciovieirarodriguez/project/NetROML/Primer problema de optimización/graphs/' + str(band[b5g]) + '_' + str(building_id) + '/'
    file_name = 'synthetic_graphs.pkl'
    with open(path + file_name, 'rb') as archivo:
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
    sum_rate = torch.sum(rates, dim=1)  
    return sum_rate

def power_constraint_per_ap(phi, pmax_per_ap):
    sum_phi_per_link = torch.sum(phi, dim=2)  
    return (sum_phi_per_link - pmax_per_ap)

def mu_update_per_ap(mu_k, power_constr_per_ap, eps):
    mu_k = mu_k.detach()
    mu_k_update = eps * torch.mean(power_constr_per_ap, dim=0)  # Promedio sobre batch: [num_links]
    mu_k = mu_k + mu_k_update
    mu_k = torch.max(mu_k, torch.tensor(0.0))
    return mu_k


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
    for c in range(num_channels):
        if c > 0:
            interf_overlap[:,:,c] += torch.einsum('bij,bj->bi', channel_matrix_batch.float(), phi[:,:,c-1].float())
        if c < num_channels - 1:
            interf_overlap[:,:,c] += torch.einsum('bij,bj->bi', channel_matrix_batch.float(), phi[:,:,c+1].float())
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
        
        # CORREGIDO: mantener SNR válido, anular inválido
        snr[:, :, c] = snr[:, :, c] * (~invalid).float()  # ~ para negar

    # Tasa por enlace
    rates = torch.sum(torch.log1p(snr), dim=-1) # [batch, links]
    return rates