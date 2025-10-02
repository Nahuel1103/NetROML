# from torch_geometric.nn import LayerNorm, Sequential
# from torch_geometric.nn.conv import MessagePassing

import random
import pickle
import numpy as np
# import networkx as nx

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import grad

# import torch_geometric as pyg
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import TAGConv
# from torch_geometric.nn import GCNConv
# import matplotlib.pyplot as plt



def nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=4, alpha=0.3):
    """
    phi: [batch_size, num_links, num_channels] (potencia por canal)
    channel_matrix_batch: [batch_size, num_links, num_links] (ganancias |h_ji|^2)
    sigma: ruido (scalar o tensor compatible)
    p0: potencia máxima por canal
    alpha: factor de interferencia para canales solapados (0 <= alpha <= 1)
    """
    batch_size, num_links, num_channels = phi.shape

    # señal útil
    diagH = torch.diagonal(channel_matrix_batch, dim1=1, dim2=2)  # [batch, links]
    signal = diagH.unsqueeze(-1) * phi  # [batch, links, channels]

    # interferencia intra-canal (misma frecuencia)
    interf_same = torch.einsum('bij,bjc->bic', channel_matrix_batch.float(), phi.float()) - signal

    # interferencia por canales solapados (vecinos inmediatos)
    interf_overlap = torch.zeros_like(interf_same)
    for c in range(num_channels):
        if c > 0:
            interf_overlap[:,:,c] += torch.einsum('bij,bj->bi', channel_matrix_batch.float(), phi[:,:,c-1].float())
        if c < num_channels - 1:
            interf_overlap[:,:,c] += torch.einsum('bij,bj->bi', channel_matrix_batch.float(), phi[:,:,c+1].float())
    interf_overlap *= alpha

    # denominador total
    denom = sigma + interf_same + interf_overlap

    # SINR
    snr = signal / denom

    # tasa por enlace
    rates = torch.sum(torch.log1p(snr), dim=-1)  # [batch, links]

    return rates



def compute_reward(phi, channel_matrix_batch, sigma, p0=4, alpha=0.3, lambda_p=0.01):
    rates = nuevo_get_rates(phi, channel_matrix_batch, sigma, p0, alpha)  # [batch, links]
    throughput = torch.sum(rates, dim=1)  # [batch] suma de todos los enlaces
    power_penalty = lambda_p * torch.sum(phi, dim=(1,2))  # [batch]
    reward = throughput - power_penalty
    return reward  # [batch]



def get_reward(mu_power,power_history, Pmax, phi, channel_matrix_batch, sigma, p0=4, alpha=0.3):
    # cambiar este mu
    mu=0.5
    
    avg_power = np.mean(power_history, axis=0)   # [nAPs,1]

    power_penalty_per_AP = avg_power - Pmax
    power_penalty=np.sum(power_penalty_per_AP)

    all_rates=nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=4, alpha=0.3)
    rate = torch.sum(all_rates,dim=1)
    
    reward = rate - mu_power*power_penalty