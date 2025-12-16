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

        if H.dim() == 2:
            H = H.unsqueeze(0)
            mu = mu.unsqueeze(0)

        batch_size, n_APs, _ = H.shape
        device = H.device

        outputs = []

        for i in range(batch_size):
            # Node features: [1, mu_i]
            x = torch.ones(n_APs, 1, device=device)
            x = torch.cat([x, mu[i].unsqueeze(-1)], dim=1)  # [n_APs, 2]

            # Estructura del grafo
            edge_index = H[i].nonzero(as_tuple=False).t()
            edge_attr = H[i][edge_index[0], edge_index[1]]

            # Normalización de edge_attr (importante para TAGConv con pesos grandes)
            norm = torch.norm(H[i], p=2)
            norm = norm if norm > 1e-12 else 1.0
            edge_attr = edge_attr / norm

            # Pasar por GNN → [n_APs, hidden_dim]
            gnn_out = self.gnn(x, edge_index, edge_attr)
            outputs.append(gnn_out)


        # Apilamos resultados
        # [batch_size, n_APs, hidden_dim]
        return torch.stack(outputs, dim=0)


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

        # Desactivar la capa interna que SB3 agregaba por defecto
        self.action_net = nn.Identity()

        # Sobrescribimos optimizador
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1))

    
    def forward(self, obs, deterministic=False):
        """
        Genera acciones, valores y log-probs a partir de observaciones.
        """
        features_per_node = self.features_extractor(obs)       # [batch_size, n_APs, hidden_dim]

        batch_size = features_per_node.shape[0]

        # logits_per_node = self.actor_head(features_per_node)   # [batch_size, n_APs, n_actions]
        # value_per_node = self.value_head(features_per_node)    # [batch_size, n_APs, 1]

        # Reshape para aplicar linear layer: [batch*n_APs, hidden_dim]
        features_flat = features_per_node.reshape(-1, self.hidden_dim)

        # Aplicar heads
        logits_flat = self.actor_head(features_flat)  # [batch*n_APs, n_actions]
        value_flat = self.value_head(features_flat)   # [batch*n_APs, 1]

        # Reshape back: [batch, n_APs, n_actions] y [batch, n_APs, 1]
        logits_per_node = logits_flat.reshape(batch_size, self.n_APs, self.num_actions_per_node)
        value_per_node = value_flat.reshape(batch_size, self.n_APs, 1)

        # ✅ CLAVE: Concatenar logits de todos los nodos en dim=2
        # [batch, n_APs * n_actions]
        logits = logits_per_node.reshape(batch_size, -1)
        
        # Valor global del grafo
        value = value_per_node.mean(dim=1)  # [batch, 1]

        # # PPO espera logits aplanados
        # # logits = logits_per_node.flatten()
        # logits = logits_per_node.reshape(batch_size, -1)
        # value = value_per_node.mean(dim=1).reshape(batch_size, 1)     # valor global del grafo
        
        # Distribución MultiCategorical (una por nodo)
        distribution = self._get_action_dist_from_latent(logits) 

        # Muestreamos o tomamos acción determinística
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions) # SB3 espera [batch], no [batch, 1]

        return actions, value, log_probs

    def get_distribution(self, obs: torch.Tensor) -> any:
        """
        Obtiene la distribución de acciones para una observación dada.
        Necesario para extraer probabilidades manualmente.
        """
        features_per_node = self.features_extractor(obs)
        batch_size = features_per_node.shape[0]
        
        features_flat = features_per_node.reshape(-1, self.hidden_dim)
        logits_flat = self.actor_head(features_flat)
        
        logits_per_node = logits_flat.reshape(batch_size, self.n_APs, self.num_actions_per_node)
        logits = logits_per_node.reshape(batch_size, -1)
        
        return self._get_action_dist_from_latent(logits)
    

    def predict_values(self, obs):
        """
        Sobrescribe el método original de SB3.
        Devuelve un tensor 1D (un valor por entorno) como PPO espera.
        """
        features_per_node = self.features_extractor(obs)       # [batch, n_APs, hidden_dim]
        value_per_node = self.value_head(features_per_node)    # [batch, n_APs, 1]
        value = value_per_node.mean(dim=1).reshape(-1)         # [batch]
        return value



    def _predict(self, observation, deterministic=False):
        logits, _ = self.forward(observation, deterministic)
        distribution = self._get_action_dist_from_latent(logits, None)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions


    def evaluate_actions(self, obs, actions):
        """
        🎯 CRÍTICO: PPO llama este método durante el training.
        Evalúa acciones para el update de PPO.
        Devuelve: values, log_probs, entropy
        """
        features_per_node = self.features_extractor(obs)  # [batch, n_APs, hidden_dim]
        batch_size = features_per_node.shape[0]
        
        # Flatten y aplicar heads (misma lógica que forward)
        features_flat = features_per_node.reshape(-1, self.hidden_dim)
        logits_flat = self.actor_head(features_flat)  # [batch*n_APs, n_actions]
        value_flat = self.value_head(features_flat)   # [batch*n_APs, 1]
        
        # Reshape back
        logits_per_node = logits_flat.reshape(batch_size, self.n_APs, self.num_actions_per_node)
        value_per_node = value_flat.reshape(batch_size, self.n_APs, 1)
        
        # Logits concatenados para SB3
        logits = logits_per_node.reshape(batch_size, -1)  # [batch, n_APs * n_actions]
        value = value_per_node.mean(dim=1).squeeze(-1)     # [batch]
        
        # Distribución y evaluación
        distribution = self._get_action_dist_from_latent(logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return value, log_probs, entropy