"""
Policy personalizada GNN para NetworkEnvironment
Compatible con observation space: {'channel_matrix', 'mu_k'}
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gnn import GNN


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extractor de features usando GNN
    Procesa channel_matrix como grafo y mu_k como features de nodos
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        gnn_hidden_dim: int = 32,
        gnn_num_layers: int = 3,
        K: int = 3,
        input_dim: int = 2,  # [bias=1, mu_k]
    ):
        super().__init__(observation_space, features_dim=gnn_hidden_dim)

        self.gnn = GNN(
            input_dim=input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=False,
            K=K
        )
        self.gnn_hidden_dim = gnn_hidden_dim

    def forward(self, observations: dict) -> torch.Tensor:
        """
        observations: dict con 'channel_matrix' y 'mu_k'
        Returns: [batch, num_links, hidden_dim]
        """
        # Usar nombres correctos del NetworkEnvironment
        H = observations["channel_matrix"]
        mu = observations["mu_k"]

        # Asegurar dimensión de batch
        if H.dim() == 2:
            H = H.unsqueeze(0)
            mu = mu.unsqueeze(0)

        batch_size, num_links, _ = H.shape
        device = H.device

        outputs = []

        for i in range(batch_size):
            # Node features: [bias=1, mu_k_i]
            x = torch.ones(num_links, 1, device=device)
            x = torch.cat([x, mu[i].unsqueeze(-1)], dim=1)  # [num_links, 2]

            # Construir edge_index y edge_attr desde channel_matrix
            edge_index = H[i].nonzero(as_tuple=False).t()
            
            # Manejar grafos vacíos o matriz de fin (-1)
            if edge_index.shape[1] == 0 or (H[i] < 0).any():
                # Self-loops con peso pequeño
                edge_index = torch.arange(num_links, device=device).unsqueeze(0).repeat(2, 1)
                edge_attr = torch.ones(num_links, device=device) * 1e-6
            else:
                edge_attr = H[i][edge_index[0], edge_index[1]]

            # Pasar por GNN
            gnn_out = self.gnn(x, edge_index, edge_attr)
            outputs.append(gnn_out)

        return torch.stack(outputs, dim=0)  # [batch, num_links, hidden_dim]


class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Policy Actor-Critic con GNN para NetworkEnvironment
    Compatible con MultiDiscrete action space
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.MultiDiscrete,
        lr_schedule,
        gnn_hidden_dim: int = 32,
        gnn_num_layers: int = 3,
        K: int = 3,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_num_layers=gnn_num_layers,
                K=K
            ),
            net_arch=None,
            *args,
            **kwargs
        )

        self.num_actions_per_node = action_space.nvec[0]
        self.num_links = len(action_space.nvec)
        self.hidden_dim = gnn_hidden_dim

        # Heads independientes por nodo
        self.actor_head = nn.Linear(self.hidden_dim, self.num_actions_per_node)
        self.value_head = nn.Linear(self.hidden_dim, 1)

        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1))

    
    def forward(self, obs, deterministic=False):
        """
        Genera acciones, valores y log-probs
        
        Returns:
            actions: [batch, num_links]
            values: [batch]
            log_probs: [batch]
        """
        features_per_node = self.features_extractor(obs)  # [batch, num_links, hidden_dim]
        batch_size = features_per_node.shape[0]

        # Flatten para aplicar linear layers
        features_flat = features_per_node.reshape(-1, self.hidden_dim)
        logits_flat = self.actor_head(features_flat)  # [batch*num_links, num_actions]
        value_flat = self.value_head(features_flat)   # [batch*num_links, 1]

        # Reshape back
        logits_per_node = logits_flat.reshape(batch_size, self.num_links, self.num_actions_per_node)
        value_per_node = value_flat.reshape(batch_size, self.num_links, 1)

        # Concatenar logits para MultiDiscrete
        logits = logits_per_node.reshape(batch_size, -1)  # [batch, num_links * num_actions]
        
        # Valor global: promedio de valores por nodo
        value = value_per_node.mean(dim=1).squeeze(-1)  # [batch]

        # Distribución y muestreo
        distribution = self._get_action_dist_from_latent(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        
        # Asegurar shape correcto [batch]
        if log_probs.dim() > 1:
            log_probs = log_probs.squeeze(-1)

        return actions, value, log_probs
    

    def predict_values(self, obs):
        """
        Predice valores del critic
        
        Returns: [batch]
        """
        features_per_node = self.features_extractor(obs)  # [batch, num_links, hidden_dim]
        features_flat = features_per_node.reshape(-1, self.hidden_dim)
        value_flat = self.value_head(features_flat)  # [batch*num_links, 1]
        value_per_node = value_flat.reshape(features_per_node.shape[0], self.num_links, 1)
        return value_per_node.mean(dim=1).squeeze(-1)  # [batch]


    def evaluate_actions(self, obs, actions):
        """
        Evalúa acciones para PPO update
        
        Returns:
            values: [batch]
            log_probs: [batch]
            entropy: [batch]
        """
        features_per_node = self.features_extractor(obs)  # [batch, num_links, hidden_dim]
        batch_size = features_per_node.shape[0]
        
        # Flatten y aplicar heads
        features_flat = features_per_node.reshape(-1, self.hidden_dim)
        logits_flat = self.actor_head(features_flat)
        value_flat = self.value_head(features_flat)
        
        # Reshape
        logits_per_node = logits_flat.reshape(batch_size, self.num_links, self.num_actions_per_node)
        value_per_node = value_flat.reshape(batch_size, self.num_links, 1)
        
        # Logits concatenados y valor global
        logits = logits_per_node.reshape(batch_size, -1)
        value = value_per_node.mean(dim=1).squeeze(-1)
        
        # Distribución y evaluación
        distribution = self._get_action_dist_from_latent(logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # ✅ Asegurar shapes correctos
        if log_probs.dim() > 1:
            log_probs = log_probs.squeeze(-1)
        if entropy.dim() > 1:
            entropy = entropy.squeeze(-1)
        
        return value, log_probs, entropy