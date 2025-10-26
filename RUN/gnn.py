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


#######################################################
### Pytorch Geometric GNN model##
#######################################################
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, K=1):
      super(GNN, self).__init__()

      self.num_layers = num_layers
      self.hidden_dim = hidden_dim
      self.dropout = dropout
      self.convs = torch.nn.ModuleList()
      self.K = K

      # first layer
      self.convs.append(TAGConv(in_channels = input_dim, out_channels = hidden_dim, K = K, bias = True, normalize = False))
      # intermediate layers
      for _ in range(num_layers - 2):
        self.convs.append(TAGConv(in_channels = hidden_dim, out_channels = hidden_dim, K = K, bias = True, normalize = False)) 
      # last layer
      self.convs.append(TAGConv(in_channels = hidden_dim, out_channels = output_dim, K = K, bias = False, normalize = False))

      self.initialize_weights()

    def initialize_weights(self):
      for name, param in self.convs.named_parameters():
        if 'weight' in name:
         nn.init.normal_(param.data, mean=0.0, std=0.1)
        elif 'bias' in name:
          nn.init.constant_(param.data, 0.1)

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers):
            x = self.convs[i](x = x, edge_index = edge_index, edge_weight = edge_attr)
            if (i  < (self.num_layers -1)):
                x = F.leaky_relu(x, inplace = False)
        return x

#######################################################