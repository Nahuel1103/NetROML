"""
Policy personalizada que integra GNN con Stable-Baselines3
SIN BATCHES - procesa un grafo a la vez
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gnn import GNN


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extrae features por nodo desde un grafo (H, mu) usando una GNN.
    Devuelve una representación [n_APs, hidden_dim] independiente del número de nodos.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        gnn_hidden_dim: int = 32,
        gnn_num_layers: int = 3,
        K: int = 3,
        input_dim: int = 2,  # [1 base + mu_i]
    ):
        # En SB3, debemos especificar un features_dim "falso",
        # pero en este caso no se usa porque devolvemos features por nodo
        super().__init__(observation_space, features_dim=gnn_hidden_dim)

        self.gnn = GNN(
            input_dim=input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=False,
            K=K
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """
        observations: dict con keys 'H' y 'mu'
        Devuelve: tensor [n_APs, hidden_dim]
        """
        H = observations["H"]
        mu = observations["mu"]

        # Convertir a batch si no lo es
        if H.dim() == 3:
            H = H[0]
            mu = mu[0]

        n_APs = H.shape[0]
        device = H.device

        # Node features: [1, mu_i]
        x = torch.ones(n_APs, 1, device=device)
        x = torch.cat([x, mu.unsqueeze(-1)], dim=1)  # [n_APs, 2]

        # Estructura del grafo
        edge_index = H.nonzero(as_tuple=False).t()
        edge_attr = H[edge_index[0], edge_index[1]]

        # Pasar por GNN → [n_APs, hidden_dim]
        gnn_out = self.gnn(x, edge_index, edge_attr)
        return gnn_out  # no se aplana, se conserva por nodo


class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Policy Actor-Critic que genera una distribución independiente por nodo.
    Compatible con MultiDiscrete([n_actions_por_nodo] * n_APs)
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
            net_arch=None,  # usamos heads personalizadas
            *args,
            **kwargs
        )

        # Configuración del entorno
        self.num_actions_per_node = action_space.nvec[0]
        self.n_APs = len(action_space.nvec)
        self.hidden_dim = gnn_hidden_dim

        # Actor y Critic "por nodo"
        self.actor_head = nn.Linear(self.hidden_dim, self.num_actions_per_node)
        self.value_head = nn.Linear(self.hidden_dim, 1)

        # 🔧 Desactivar la capa interna que SB3 agregaba por defecto
        self.action_net = nn.Identity()

        # Sobrescribimos optimizador
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1))

    
    def forward(self, obs, deterministic=False):
        """
        Genera acciones, valores y log-probs a partir de observaciones.
        """
        features_per_node = self.features_extractor(obs)       # [n_APs, hidden_dim]
        logits_per_node = self.actor_head(features_per_node)   # [n_APs, n_actions]
        value_per_node = self.value_head(features_per_node)    # [n_APs, 1]

        # PPO espera logits aplanados
        logits = logits_per_node.flatten()
        # value = value_per_node.mean()  # valor global del grafo
        # value = value_per_node.mean().view(1, 1)
        value = value_per_node.mean().reshape(1, 1)
        
        print(value)


        # Distribución MultiCategorical (una por nodo)
        distribution = self._get_action_dist_from_latent(logits.unsqueeze(0)) #el unsqueeze es porque SB3 espera batch

        # Muestreamos o tomamos acción determinística
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions).unsqueeze(1) #unsqueeze para que tenga shape [batch, 1]

        return actions, value, log_probs


    def _predict(self, observation, deterministic=False):
        logits, _ = self.forward(observation, deterministic)
        distribution = self._get_action_dist_from_latent(logits, None)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions
