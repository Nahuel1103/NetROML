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

def transform_matrix(adj_matrix, all=True):
    # mantuve tu lógica funcional pero podés vectorizar más tarde
    if all:
        nodos = adj_matrix.shape[0]
        lista_parejas = []
        for i in range(nodos):
            receptor = np.argmax(adj_matrix[i, :])
            valor_maximo = adj_matrix[i, receptor]
            transmisor = i
            elemento = [valor_maximo, transmisor, receptor]
            lista_parejas.append(elemento)

        H = np.zeros_like(adj_matrix)
        for i in range(nodos):
            for j in range(nodos):
                if i == j:
                    H[i, j] = lista_parejas[i][0]
                else:
                    # busco el receptor asociado al transmisor j
                    nodo_receptor = 0
                    for k in range(nodos):
                        if lista_parejas[k][1] == j:
                            nodo_receptor = lista_parejas[k][2]
                            break
                    if nodo_receptor == i:
                        H[i, j] = 0.005
                    else:
                        H[i, j] = adj_matrix[i, nodo_receptor]
        return H
    else:
        # versión partial (preservando tu lógica)
        num_nodes = adj_matrix.shape[0]
        nodos = list(np.arange(num_nodes))
        random.seed(42)
        random.shuffle(nodos)
        half_nodes = num_nodes // 2
        nodos_tx = nodos[:half_nodes]
        nodos_rx = nodos[half_nodes:]
        lista_parejas = []
        for nodo_tx in nodos_tx:
            h_canal = [adj_matrix[nodo_tx, nodo_rx] for nodo_rx in nodos_rx]
            index_pareja = np.argmax(h_canal)
            valor_maximo = adj_matrix[nodo_tx, nodos_rx[index_pareja]]
            lista_parejas.append([valor_maximo, nodo_tx, nodos_rx[index_pareja]])

        H = np.zeros((num_nodes, num_nodes))
        for i in range(half_nodes):
            for j in range(half_nodes):
                if i == j:
                    H[i, j] = lista_parejas[i][0]
                else:
                    nodo_tx_a_j = lista_parejas[j][1]
                    receptor_j = None
                    for k in range(half_nodes):
                        if lista_parejas[k][1] == nodo_tx_a_j:
                            receptor_j = k
                            break
                    H[i, j] = adj_matrix[nodos_tx[i], nodos_rx[receptor_j]]
        return H

    
def graphs_to_tensor(train=True, num_links=5, num_features=1, b5g=False, building_id=990, base_path=None):
    band = ['2_4', '5']
    if base_path is None:
        path = '/Users/mauriciovieirarodriguez/project/NetROML/graphs/' + str(band[b5g]) + '_' + str(building_id) + '/'
    else:
        path = base_path
    file_name = ('train_' if train else 'val_') + f"{band[b5g]}_graphs_{building_id}.pkl"
    with open(path + file_name, 'rb') as archivo:
        graphs = pickle.load(archivo)

    x_list = []
    channel_matrix_list = []
    x = torch.zeros((num_links, num_features), dtype=torch.float)
    for graph in graphs:
        adj_matrix = nx.adjacency_matrix(graph, weight='Atenuacion').toarray()
        channel_matrix = transform_matrix(adj_matrix, all=True)
        # tu escalado: channel_matrix / 1e-3 -> convertir a float y tensor
        channel_matrix_list.append(torch.tensor(channel_matrix.T, dtype=torch.float))
        x_list.append(x.clone())

    channel_matrix_tensor = torch.stack(channel_matrix_list)  # [batch, num_channels, num_links]? (vos usás transpuesta)
    x_tensor = torch.stack(x_list)  # [batch, num_links, num_features]
    return x_tensor, channel_matrix_tensor


def graphs_to_tensor_synthetic(num_links, num_features = 1, b5g = False, building_id = 990):
    
    band = ['2_4', '5']
    path = '/Users/mauriciovieirarodriguez/project/NetROML/graphs/' + str(band[b5g]) + '_' + str(building_id) + '/'
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

def get_gnn_inputs(x_tensor, channel_matrix_tensor, normalize=True, eps=1e-12):
    """
    Inputs:
      x_tensor: [batch, num_nodes, num_features]  (aquí num_nodes probablemente = num_links o num_channels)
      channel_matrix_tensor: [batch, N, N]  (modelo: H matrices)
    Output:
      list de torch_geometric.data.Data con:
        - data.x: [N, num_features]
        - data.edge_index: [2, num_edges]
        - data.edge_attr: [num_edges]
        - data.matrix: la matrix original (opcional)
    """
    input_list = []
    size = channel_matrix_tensor.shape[0]  # batch size
    for i in range(size):
        x = x_tensor[i, :, :].detach().clone()
        channel_matrix = channel_matrix_tensor[i, :, :].detach().clone().float()

        if normalize:
            norm = torch.norm(channel_matrix, p=2)
            norm = norm if norm > eps else 1.0
            channel_matrix_norm = channel_matrix / norm
        else:
            channel_matrix_norm = channel_matrix

        # edge_index: indices donde H != 0  -> devuelve [num_edges, 2] si as_tuple=False
        nz = (channel_matrix_norm != 0).nonzero(as_tuple=False)  # [[i,j],...]
        if nz.numel() == 0:
            # grafo vacío: evitamos crash creando un self-loop pequeño
            N = channel_matrix_norm.shape[0]
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([channel_matrix_norm[0, 0]], dtype=torch.float)
        else:
            edge_index = nz.t().contiguous().long()  # [2, num_edges]
            edge_attr = channel_matrix_norm[edge_index[0], edge_index[1]].to(torch.float)  # [num_edges]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, matrix=channel_matrix)
        input_list.append(data)
    return input_list

def load_channel_matrix(building_id, b5g, num_links, synthetic=False, shuffle=True, repeat=True):
    """Carga el dataset de matrices de canal."""
    print(f"Cargando dataset (building_id={building_id}, synthetic={synthetic})...")
    
    if synthetic:
        x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
            num_links=num_links,
            num_features=1,
            b5g=b5g,
            building_id=building_id
        )
        # dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        channel_matrix_tensor = channel_matrix_tensor[:7000]
    else:
        x_tensor, channel_matrix_tensor = graphs_to_tensor(
            train=True,
            num_links=num_links,
            num_features=1,
            b5g=b5g,
            building_id=building_id
        )
        # dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
    
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Convertir a numpy
    matrices = channel_matrix_tensor.numpy().astype(np.float32)
    indices = np.arange(len(matrices))
    
    print(f"✓ Iterador creado: {len(matrices)} matrices")
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            yield matrices[idx]
        
        if not repeat:
            break  

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
    """
    phi: [batch, num_links, num_channels]  (potencias por enlace por canal)
    channel_matrix_batch: [batch, num_links, num_links] (H matrices según tu transform_matrix)
    sigma: scalar o tensor [batch, 1]
    return: rates [batch, num_links]
    """
    batch_size, num_links, num_channels = phi.shape

    # Señal útil: diagH shape [batch, num_links]
    diagH = torch.diagonal(channel_matrix_batch, dim1=1, dim2=2)  # [batch, num_links]
    signal = diagH.unsqueeze(-1) * phi  # [batch, num_links, num_channels]

    # Interferencia intra-canal (misma canal)
    interf_same = torch.einsum('bij,bjc->bic', channel_matrix_batch.float(), phi.float()) - signal  # [batch, links, channels]

    # Interferencia por canales solapados (evitando wrap-around)
    interf_overlap = torch.zeros_like(interf_same)
    if num_channels > 1:
        # creamos left and right con padding en extremos (no wrap)
        left = torch.zeros_like(phi)
        right = torch.zeros_like(phi)
        left[:, :, 1:] = phi[:, :, :-1]   # left shift (canal c sees c-1)
        right[:, :, :-1] = phi[:, :, 1:]  # right shift (canal c sees c+1)
        interf_overlap = alpha * (left + right)  # solo un alpha

    denom = sigma + interf_same + interf_overlap  # broadcasting con sigma posible
    snr = signal / (denom + eps)  # [batch, links, channels]

    # invalidación por recepciones múltiples fuertes en mismo canal
    for c in range(num_channels):
        p_ch = phi[:, :, c]  # [batch, links]
        # recv_power: para cada par (i,j) = H_ij * p_jc
        # channel_matrix_batch: [batch, links, links]
        recv_power = channel_matrix_batch * p_ch.unsqueeze(1)  # [batch, links, links]
        tx_active = p_ch > eps  # [batch, links]
        # seen: receptor i ve potencia de algún tx j en ese canal con p_rx_threshold
        seen = (recv_power >= p_rx_threshold) & (tx_active.unsqueeze(1))
        # seen_count por receptor (cuantos tx alcanzan receptor i)
        seen_count = seen.sum(dim=-1)  # [batch, links] (sum sobre transmisores j)
        invalid = seen_count >= 2  # [batch, links]
        # apago SNR en canal c para enlaces 'invalid'
        snr[:, :, c] = snr[:, :, c] * (~invalid).float()

    # tasa por enlace: sumo SNR sobre canales y aplico log1p
    rates = torch.log1p(torch.sum(snr, dim=-1))  # [batch, links]
    return rates