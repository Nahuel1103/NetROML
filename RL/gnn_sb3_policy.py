"""
Integración de GNN con Stable-Baselines3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


#######################################################
### GNN Policy Original (como feature extractor)
#######################################################
class GNNPolicy(torch.nn.Module):
    """
    Extrae features de la red.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, K=1):

        super(GNNPolicy, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.K = K

        # First layer
        self.convs.append(TAGConv(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            K=K, 
            bias=True, 
            normalize=False
        ))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim, 
                K=K, 
                bias=True, 
                normalize=False
            ))
        
        # Last layer: output features (no acciones directas)
        self.convs.append(TAGConv(
            in_channels=hidden_dim, 
            out_channels=output_dim, 
            K=K, 
            bias=True,  # Changed to True para features
            normalize=False
        ))

        self.initialize_weights()

    def initialize_weights(self):

        for name, param in self.convs.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.1)

    def forward(self, x, edge_index, edge_attr):

        h = x
        for i in range(self.num_layers):
            h = self.convs[i](x=h, edge_index=edge_index, edge_weight=edge_attr)
            if i < (self.num_layers - 1):
                h = F.leaky_relu(h, inplace=False)
        return h


#######################################################
### Feature Extractor para SB3
#######################################################
class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Wrapper de GNNPolicy como feature extractor para Stable-Baselines3.
    
    FLUJO:
    1. Recibe observación plana (ignorada)
    2. Usa graph_data inyectado externamente
    3. Procesa con GNNPolicy
    4. Retorna features para actor-critic
    """
    
    def __init__(self, 
                 observation_space: spaces.Box,
                 num_links: int = 5,
                 hidden_dim: int = 16,
                 num_layers: int = 5,
                 K: int = 3,
                 features_dim: int = 64):
        """
        Args:
            observation_space: Espacio de observación (requerido por SB3)
            num_links: Número de enlaces/nodos en el grafo
            hidden_dim: Dimensión oculta de TAGConv
            num_layers: Número de capas convolucionales
            K: Número de saltos en TAGConv
            features_dim: Dimensión final de features (debe ser num_links * hidden_dim)
        """
        # Features dim debe coincidir con la salida del GNN
        actual_features_dim = num_links * hidden_dim
        super().__init__(observation_space, features_dim=actual_features_dim)
        
        self.num_links = num_links
        self.hidden_dim = hidden_dim
        self._features_dim = actual_features_dim
        
        # Crear GNNPolicy como feature extractor
        # input_dim=1 (solo mu_k), output_dim=hidden_dim (features por nodo)
        self.gnn = GNNPolicy(
            input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=False,
            K=K
        )
        
        # Cache para graph_data
        self.current_graph_data = None

    def set_graph_data(self, graph_data):
        """
        Inyecta los datos del grafo para el próximo forward pass.
        Llamado externamente por la política o el callback.
        """
        self.current_graph_data = graph_data

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Procesa el grafo con GNNPolicy y retorna features.
        
        Args:
            observations: [batch_size, obs_dim] - Observaciones planas (IGNORADAS)
        
        Returns:
            features: [batch_size, num_links * hidden_dim] - Features para actor-critic
        """
        if self.current_graph_data is None:
            # Fallback: retornar ceros si no hay graph_data
            # (solo durante inicialización)
            batch_size = observations.shape[0]
            return torch.zeros(
                batch_size, 
                self._features_dim, 
                device=observations.device
            )
        
        # Extraer datos del grafo
        graph_data = self.current_graph_data
        x = graph_data.x              # [num_links, 1]
        edge_index = graph_data.edge_index  # [2, num_edges]
        edge_attr = graph_data.edge_attr    # [num_edges]
        
        # Forward pass por GNNPolicy
        node_embeddings = self.gnn(x, edge_index, edge_attr)
        # node_embeddings shape: [num_links, hidden_dim]
        
        # Flatten para obtener features globales
        features_flat = node_embeddings.flatten()  # [num_links * hidden_dim]
        
        # Expandir para batch (SB3 espera batch dimension)
        batch_size = observations.shape[0]
        features = features_flat.unsqueeze(0).expand(batch_size, -1)
        
        return features


#######################################################
### Actor-Critic Policy para SB3
#######################################################
class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Política Actor-Critic que usa GNNFeaturesExtractor.
    
    ARQUITECTURA:
    graph_data → GNNPolicy → [num_links * hidden_dim] features
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
                  Actor                  Critic
                (MLP 64→64)            (MLP 64→64)
                    ↓                       ↓
              Action logits              Value
    """
    
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule,
                 num_links: int = 5,
                 gnn_hidden_dim: int = 16,
                 gnn_num_layers: int = 5,
                 gnn_K: int = 3,
                 **kwargs):
        
        self.num_links = num_links
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_layers = gnn_num_layers
        self.gnn_K = gnn_K
        
        # Features dimension = num_links * hidden_dim
        features_dim = num_links * gnn_hidden_dim
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs={
                'num_links': num_links,
                'hidden_dim': gnn_hidden_dim,
                'num_layers': gnn_num_layers,
                'K': gnn_K,
                'features_dim': features_dim
            },
            **kwargs
        )
        
        # Cache para graph_data (será inyectado externamente)
        self.current_graph_data = None

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Override: Inyecta graph_data antes de extraer features.
        """
        if self.current_graph_data is not None:
            self.features_extractor.set_graph_data(self.current_graph_data)
        
        return super().extract_features(obs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass completo de la política.
        Asegura que graph_data esté disponible.
        """
        return super().forward(obs, deterministic)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predice valores para el critic.
        Override para inyectar graph_data.
        """
        if self.current_graph_data is not None:
            self.features_extractor.set_graph_data(self.current_graph_data)
        
        return super().predict_values(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evalúa acciones para el entrenamiento.
        Override para inyectar graph_data.
        """
        if self.current_graph_data is not None:
            self.features_extractor.set_graph_data(self.current_graph_data)
        
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs: torch.Tensor):
        """
        Obtiene la distribución de acciones.
        Override para inyectar graph_data.
        """
        if self.current_graph_data is not None:
            self.features_extractor.set_graph_data(self.current_graph_data)
        
        return super().get_distribution(obs)


#######################################################
### Función helper para crear la política
#######################################################
def create_gnn_policy_kwargs(num_links=5, gnn_hidden_dim=16, gnn_num_layers=5, 
                             gnn_K=3, net_arch=None):
    """
    Crea el diccionario de configuración para GNNActorCriticPolicy.
    
    Args:
        num_links: Número de enlaces en la red
        gnn_hidden_dim: Dimensión oculta de la GNN
        gnn_num_layers: Número de capas TAGConv
        gnn_K: Número de saltos en TAGConv
        net_arch: Arquitectura del actor-critic después de la GNN
    
    Returns:
        policy_kwargs: Diccionario para pasar a PPO
    
    Uso:
        policy_kwargs = create_gnn_policy_kwargs(num_links=5, gnn_hidden_dim=16)
        model = PPO(GNNActorCriticPolicy, env, policy_kwargs=policy_kwargs)
    """
    if net_arch is None:
        # Arquitectura por defecto: 2 capas de 64 neuronas
        net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    
    return {
        'num_links': num_links,
        'gnn_hidden_dim': gnn_hidden_dim,
        'gnn_num_layers': gnn_num_layers,
        'gnn_K': gnn_K,
        'net_arch': net_arch,
        'activation_fn': torch.nn.ReLU
    }