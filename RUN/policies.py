# ==================== policies.py ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNNExtractor(BaseFeaturesExtractor):
    """
    Extrae features de grafos usando TAGConv para reemplazar la MLP en SB3.
    - Input: observation (channel_matrix) [batch, num_links, num_links]
    - Output: embedding vectorial para la política (features_dim)
    """

    def __init__(self, observation_space, hidden_dim=1, num_layers=3):
        # Llamamos al init de BaseFeaturesExtractor con el tamaño de salida
        super().__init__(observation_space, features_dim=hidden_dim)

        # la obs puede venir aplanada (num_links*num_links,) o 2D (num_links, num_links)
        if len(observation_space.shape) == 1:
            num_links = int(observation_space.shape[0] ** 0.5)
        else:
            num_links = observation_space.shape[0]

        self.convs = nn.ModuleList()
        in_channels = num_links
        out_channels = hidden_dim

        # Creamos num_layers capas TAGConv
        for i in range(num_layers):
            self.convs.append(TAGConv(in_channels, out_channels))
            in_channels = out_channels

    def forward(self, observations):
        """
        observations: tensor [batch_size, num_links*num_links] o [batch_size, num_links, num_links]
        Representa la matriz de canal.
        """
        batch_size = observations.size(0)
        if observations.dim() == 2:
            num_links = int(observations.size(1) ** 0.5)
            obs_matrix = observations.view(batch_size, num_links, num_links)
        else:
            num_links = observations.size(1)
            obs_matrix = observations

        # construimos edge_index para un grafo totalmente conectado
        edge_index = self.build_edge_index(num_links).to(observations.device)

        # reshape: tratamos cada fila de la channel_matrix como features de un nodo
        x = obs_matrix.reshape(batch_size * num_links, -1)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # Global mean pooling: promedio sobre nodos
        x = x.reshape(batch_size, -1).mean(dim=1)

        return x

    def build_edge_index(self, num_nodes):
        """
        Construye un grafo completamente conectado sin self-loops.
        """
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index
    
    