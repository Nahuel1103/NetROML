import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_aps, K=2):
        super(GNN, self).__init__()
        self.num_aps = num_aps
        
        # TAGConv Layers (Topology Adaptive Graph Convolution)
        # K = number of hops
        self.conv1 = TAGConv(num_node_features, hidden_channels, K=K)
        self.conv2 = TAGConv(hidden_channels, hidden_channels, K=K)
        
        # Action Heads for AP Nodes
        # Channel Selection (3 channels)
        self.channel_head = nn.Linear(hidden_channels, 3) 
        # Power Selection (3 levels)
        self.power_head = nn.Linear(hidden_channels, 3)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, TAGConv):
                # PyG convs usually have their own init, but we can reset if needed
                pass

    def forward(self, x, edge_index, edge_attr):
        # edge_attr needs to be 1D weights for TAGConv usually, check signature
        # TAGConv(in, out, K) forward(x, edge_index, edge_weight=None)
        
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None
        
        x = self.conv1(x, edge_index, edge_weight)
        x = F.leaky_relu(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.leaky_relu(x)
        
        # Extract embeddings for AP nodes (first num_aps nodes)
        ap_embeddings = x[:self.num_aps]
        
        # Output Logits for REINFORCE (Categorical distribution parameters)
        channel_logits = self.channel_head(ap_embeddings)
        power_logits = self.power_head(ap_embeddings)
        
        return channel_logits, power_logits
