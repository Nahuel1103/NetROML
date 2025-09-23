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



band = ['2_4', '5']
path = '/Users/nahuelpineyro/NetROML/graphs/' + str(band[0]) + '_' + str(990) + '/'
file_name = 'synthetic_graphs.pkl'

with open(path + file_name, 'rb') as archivo:
    graphs = pickle.load(archivo)

print(f"Número de grafos: {len(graphs)}")
print(f"Tipo del primer elemento: {type(graphs[0])}")

# Examinar el primer grafo
if hasattr(graphs[0], 'shape'):
    print(f"Forma de la matriz: {graphs[0].shape}")
    print(f"Valores de ejemplo:\n{graphs[0]}")
elif isinstance(graphs[0], np.ndarray):
    print(f"Forma de la matriz: {graphs[0].shape}")
    print(f"Valores de ejemplo:\n{graphs[0]}")
else:
    print("Estructura del primer grafo:")
    print(graphs[0])

