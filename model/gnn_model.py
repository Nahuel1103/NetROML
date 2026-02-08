import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_aps, out_channels_ch=3, out_channels_pwr=4):
        super(GNN, self).__init__()
        self.num_aps = num_aps
                
        # HeteroConv Layer 1
        # ap->connects->client: Updates Client features based on AP signal (Service)
        # client->connected_to->ap: Updates AP features based on connected Client load
        self.conv1 = HeteroConv({
            ('ap', 'connects', 'client'): GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False, edge_dim=1),
            ('client', 'connected_to', 'ap'): GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
        }, aggr='sum')

        # HeteroConv Layer 2 (Deeper reasoning)
        self.conv2 = HeteroConv({
            ('ap', 'connects', 'client'): GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False, edge_dim=1),
            ('client', 'connected_to', 'ap'): GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
        }, aggr='sum')
        
        # Feature Encoders (if raw features sizes differ, which they do)
        self.ap_encoder = Linear(-1, hidden_channels)
        self.client_encoder = Linear(-1, hidden_channels)
        
        # Policy Heads (AP Only)
        # Input: AP Embedding -> Output: Action Logits
        self.channel_head = nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels), # *2 due to 2 heads
            nn.ReLU(),
            Linear(hidden_channels, out_channels_ch)
        )
        
        self.power_head = nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels_pwr)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # 1. Encode Features
        # x_dict is a dictionary active_node_type -> features
        x_dict_out = {}
        # APs always exist
        x_dict_out['ap'] = self.ap_encoder(x_dict['ap'])
        
        # Clients might not exist in empty steps
        if 'client' in x_dict:
             x_dict_out['client'] = self.client_encoder(x_dict['client'])
        
        # 2. Graph Convolutions        
        x_dict_out = self.conv1(x_dict_out, edge_index_dict, edge_attr_dict)
        x_dict_out = {key: F.elu(x) for key, x in x_dict_out.items()}
        
        x_dict_out = self.conv2(x_dict_out, edge_index_dict, edge_attr_dict)
        x_dict_out = {key: F.elu(x) for key, x in x_dict_out.items()}
        
        # 3. Decision Making (APs only)
        ap_embeddings = x_dict_out['ap']
        
        channel_logits = self.channel_head(ap_embeddings)
        power_logits = self.power_head(ap_embeddings)
        
        return channel_logits, power_logits
