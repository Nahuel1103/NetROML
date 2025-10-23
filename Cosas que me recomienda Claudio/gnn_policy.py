"""
Policy personalizada que integra tu GNN con Stable-Baselines3
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data, Batch
import numpy as np

# Importa tu GNN
# from gnn import GNN


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor que usa tu GNN para procesar el grafo
    Convierte las observaciones Dict{H, mu} en features para la policy
    """
    def __init__(
        self, 
        observation_space: spaces.Dict,
        n_APs: int = 5,
        gnn_hidden_dim: int = 32,
        gnn_num_layers: int = 3,
        K: int = 3,
        features_dim: int = 128
    ):
        # features_dim es la dimensión de salida del extractor
        super().__init__(observation_space, features_dim=features_dim)
        
        self.n_APs = n_APs
        
        # Aquí importarías tu GNN
        # self.gnn = GNN(
        #     input_dim=1,
        #     hidden_dim=gnn_hidden_dim, 
        #     output_dim=features_dim,
        #     num_layers=gnn_num_layers,
        #     dropout=False,
        #     K=K
        # )
        
        # Por ahora, un placeholder que procesa H y mu
        # REEMPLAZA esto con tu GNN real
        self.gnn_placeholder = nn.Sequential(
            nn.Linear(n_APs * n_APs + n_APs, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
        
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Procesa las observaciones y extrae features usando GNN
        
        Args:
            observations: Dict con keys 'H' y 'mu'
                H: [batch, n_APs, n_APs] - matriz de canal
                mu: [batch, n_APs] - multiplicadores de Lagrange
                
        Returns:
            features: [batch, features_dim] - features para la policy
        """
        H = observations['H']  # [batch, n_APs, n_APs]
        mu = observations['mu']  # [batch, n_APs]
        
        batch_size = H.shape[0]
        
        # OPCIÓN A: Usar tu GNN con torch_geometric
        # Necesitas convertir H a formato de grafo
        # graph_list = []
        # for i in range(batch_size):
        #     x = torch.ones(self.n_APs, 1)  # node features
        #     edge_index = H[i].nonzero(as_tuple=False).t()
        #     edge_attr = H[i][edge_index[0], edge_index[1]]
        #     graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        # 
        # batch_graph = Batch.from_data_list(graph_list)
        # gnn_output = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        # features = gnn_output.view(batch_size, -1)
        
        # OPCIÓN B: Placeholder simple (REEMPLAZAR con tu GNN)
        H_flat = H.reshape(batch_size, -1)
        combined = torch.cat([H_flat, mu], dim=1)
        features = self.gnn_placeholder(combined)
        
        return features


class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Policy Actor-Critic que usa tu GNN como feature extractor
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.MultiDiscrete,
        lr_schedule,
        n_APs: int = 5,
        gnn_hidden_dim: int = 32,
        gnn_num_layers: int = 3,
        K: int = 3,
        *args,
        **kwargs
    ):
        self.n_APs = n_APs
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_layers = gnn_num_layers
        self.K = K
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_APs=n_APs,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_num_layers=gnn_num_layers,
                K=K,
                features_dim=128
            ),
            *args,
            **kwargs
        )


class CustomCallback:
    """
    Callback para actualizar mu durante el entrenamiento
    Esto reemplaza tu mu_update manual
    """
    def __init__(self, env, eps=1e-4):
        self.env = env
        self.eps = eps
        
    def on_step(self, locals_, globals_):
        """
        Llamado después de cada step
        Aquí podrías actualizar mu si fuera necesario
        """
        # El entorno ya actualiza mu internamente en step()
        # Pero podrías hacer logging aquí
        return True
