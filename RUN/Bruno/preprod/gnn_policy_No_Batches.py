"""
Policy personalizada que integra tu GNN con Stable-Baselines3
SIN BATCHES - procesa un grafo a la vez
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data
import numpy as np
from gnn import GNN


# Importa tu GNN
# from gnn import GNN


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor que usa tu GNN para procesar el grafo
    Procesa UNA observación a la vez (sin batches)
    """
    def __init__(
        self, 
        observation_space: spaces.Dict,
        n_APs: int = 5,
        gnn_hidden_dim: int = 32,
        gnn_num_layers: int = 3,
        K: int = 3,
        features_dim: int = 64  # dimensión de salida reducida
    ):
        super().__init__(observation_space, features_dim=features_dim)
        
        self.n_APs = n_APs
        
        # OPCIÓN A: Usa tu GNN real (descomenta cuando lo integres)
        self.gnn = GNN(
            input_dim=2,  # [1 por node feature base + mu como feature adicional]
            hidden_dim=gnn_hidden_dim, 
            output_dim=features_dim // n_APs,  # output por nodo
            num_layers=gnn_num_layers,
            dropout=False,
            K=K
        )
        
        # OPCIÓN B: Placeholder MLP simple
        # Concatena H aplanado + mu y lo pasa por capas fully connected
        # input_size = n_APs * n_APs + n_APs  # H aplanado + mu
        # self.feature_net = nn.Sequential(
        #     nn.Linear(input_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, features_dim),
        #     nn.ReLU()
        # )
        
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Procesa UNA observación (sin batch dimension)
        
        Args:
            observations: Dict con keys 'H' y 'mu'
                H: [n_APs, n_APs] - matriz de canal
                mu: [n_APs] - multiplicadores de Lagrange
                
        Returns:
            features: [features_dim] - features para la policy
        """
        H = observations['H']    # [n_APs, n_APs]
        mu = observations['mu']  # [n_APs]
        
        # SB3 puede agregar batch dimension automáticamente
        # Detectamos si hay batch o no
        if H.dim() == 3:  # [batch, n_APs, n_APs]
            batch_size = H.shape[0]
            has_batch = True
        else:  # [n_APs, n_APs]
            H = H.unsqueeze(0)  # [1, n_APs, n_APs]
            mu = mu.unsqueeze(0)  # [1, n_APs]
            batch_size = 1
            has_batch = False
        
        # ============================================
        # OPCIÓN A: Usar tu GNN con torch_geometric
        # ============================================
        features_list = []
        for i in range(batch_size):
            # Crear node features: concatenar mu como feature de cada nodo
            x = torch.ones(self.n_APs, 1, device=H.device)  # base feature
            x = torch.cat([x, mu[i].unsqueeze(-1)], dim=1)  # [n_APs, 2]
            
            # Crear estructura de grafo desde H
            edge_index = H[i].nonzero(as_tuple=False).t()  # [2, num_edges]
            edge_attr = H[i][edge_index[0], edge_index[1]].unsqueeze(-1)  # [num_edges, 1]
            
            # Forward pass por GNN
            gnn_out = self.gnn(x, edge_index, edge_attr)  # [n_APs, hidden_dim]
            
            # Agregar features de todos los nodos
            features = gnn_out.flatten()  # [n_APs * hidden_dim]
            features_list.append(features)
        
        features = torch.stack(features_list)  # [batch, features_dim]

        
        # ============================================
        # OPCIÓN B: MLP simple (placeholder)
        # ============================================
        # H_flat = H.reshape(batch_size, -1)  # [batch, n_APs * n_APs]
        # combined = torch.cat([H_flat, mu], dim=1)  # [batch, n_APs*n_APs + n_APs]
        # features = self.feature_net(combined)  # [batch, features_dim]
        
        
        # Si no había batch dimension originalmente, removerla
        if not has_batch:
            features = features.squeeze(0)  # [features_dim]
        
        return features


class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Policy Actor-Critic que usa tu GNN como feature extractor
    
    Esta policy:
    1. Toma observaciones Dict{H, mu} 
    2. Extrae features con GNN
    3. Genera distribución sobre acciones MultiDiscrete
    4. Estima el value function
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
        net_arch: dict = None,  # arquitectura de actor y critic
        *args,
        **kwargs
    ):
        self.n_APs = n_APs
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_layers = gnn_num_layers
        self.K = K
        
        # Arquitectura por defecto: capas compartidas y luego actor/critic separados
        if net_arch is None:
            net_arch = dict(
                pi=[64, 64],  # actor: 2 capas de 64 unidades
                vf=[64, 64]   # critic: 2 capas de 64 unidades
            )
        
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
                features_dim=64
            ),
            net_arch=net_arch,
            *args,
            **kwargs
        )


class SingleStepPolicy(nn.Module):
    """
    ALTERNATIVA: Policy simple que NO usa SB3
    Útil si quieres mantener más control o hacer inferencia directa
    """
    def __init__(
        self,
        n_APs: int = 5,
        num_channels: int = 3,
        n_power_levels: int = 2,
        gnn_hidden_dim: int = 32,
        gnn_num_layers: int = 3,
        K: int = 3
    ):
        super().__init__()
        
        self.n_APs = n_APs
        self.num_channels = num_channels
        self.n_power_levels = n_power_levels
        
        # Número de acciones por AP
        self.num_actions = 1 + num_channels * n_power_levels
        
        # Feature extractor (tu GNN)
        # from gnn import GNN
        # self.gnn = GNN(...)
        
        # Placeholder
        input_size = n_APs * n_APs + n_APs
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor head: genera logits para cada AP
        self.actor = nn.Linear(64, n_APs * self.num_actions)
        
    def forward(self, H, mu):
        """
        Forward pass
        
        Args:
            H: [n_APs, n_APs] matriz de canal
            mu: [n_APs] multiplicadores
            
        Returns:
            action_logits: [n_APs, num_actions] logits para cada AP
        """
        # Flatten y concatenar
        H_flat = H.flatten()
        x = torch.cat([H_flat, mu], dim=0)
        
        # Encode
        features = self.encoder(x)
        
        # Actor
        logits = self.actor(features)
        logits = logits.view(self.n_APs, self.num_actions)
        
        return logits
    
    def get_action(self, H, mu, deterministic=False):
        """
        Obtiene una acción dada una observación
        
        Args:
            H: [n_APs, n_APs]
            mu: [n_APs]
            deterministic: si True, toma acción con max prob
            
        Returns:
            action: [n_APs] acciones discretas
        """
        logits = self.forward(H, mu)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        
        return action.numpy()


# ==============================================
# EJEMPLO DE USO
# ==============================================
def example_usage():
    """
    Ejemplo de cómo usar las policies
    """
    from envs import APNetworkEnv
    import numpy as np
    
    print("=" * 50)
    print("EJEMPLO 1: Policy con SB3 (recomendado)")
    print("=" * 50)
    
    # Crear dummy channel iterator
    def dummy_iterator():
        n_APs = 5
        for _ in range(100):
            H = np.random.rand(n_APs, n_APs).astype(np.float32)
            np.fill_diagonal(H, H.diagonal() * 2)
            yield H
    
    # Crear environment
    env = APNetworkEnv(
        n_APs=5,
        num_channels=3,
        P0=4,
        n_power_levels=2,
        Pmax=0.7,
        max_steps=50,
        H_iterator=dummy_iterator()
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"Observación inicial:")
    print(f"  H shape: {obs['H'].shape}")
    print(f"  mu shape: {obs['mu'].shape}")
    
    # Crear policy (requiere SB3 instalado)
    try:
        from stable_baselines3 import PPO
        
        policy_kwargs = dict(
            n_APs=5,
            gnn_hidden_dim=32,
            gnn_num_layers=3,
            K=3
        )
        
        model = PPO(
            GNNActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=0
        )
        
        # Hacer un step
        action, _states = model.predict(obs, deterministic=False)
        print(f"Acción generada: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.4f}")
        
    except ImportError:
        print("SB3 no instalado, saltando ejemplo 1")
    
    print("\n" + "=" * 50)
    print("EJEMPLO 2: Policy standalone (sin SB3)")
    print("=" * 50)
    
    # Usar policy standalone
    policy = SingleStepPolicy(
        n_APs=5,
        num_channels=3,
        n_power_levels=2
    )
    
    # Reset environment
    env_iterator = dummy_iterator()
    env2 = APNetworkEnv(
        n_APs=5,
        num_channels=3,
        P0=4,
        n_power_levels=2,
        Pmax=0.7,
        max_steps=50,
        H_iterator=env_iterator
    )
    obs, info = env2.reset()
    
    # Convertir a tensors
    H_tensor = torch.from_numpy(obs['H'])
    mu_tensor = torch.from_numpy(obs['mu'])
    
    # Get action
    action = policy.get_action(H_tensor, mu_tensor, deterministic=False)
    print(f"Acción generada: {action}")
    
    obs, reward, terminated, truncated, info = env2.step(action)
    print(f"Reward: {reward:.4f}")


if __name__ == "__main__":
    example_usage()
