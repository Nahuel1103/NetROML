import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any

from utils import graphs_to_tensor, graphs_to_tensor_synthetic, power_constraint, nuevo_get_rates

class WifiResourceEnv(gym.Env):
    """
    Entorno de asignación de recursos Wi-Fi para SB3.
    Puede usar MLP o GNN según `use_gnn`.
    """

    def __init__(
        self,
        num_links: int = 5,
        num_channels: int = 3,
        p0: float = 4.0,
        sigma: float = 1e-4,
        lambda_power: float = 1.0,
        max_steps: int = 50,
        use_synthetic: bool = True,
        building_id: int = 990,
        b5g: bool = False,
        use_gnn: bool = False,
        seed: int = 12345
    ):
        super().__init__()
        self.num_links = num_links
        self.num_channels = num_channels
        self.p0 = p0
        self.sigma = sigma
        self.lambda_power = lambda_power
        self.max_steps = max_steps
        self.use_synthetic = use_synthetic
        self.building_id = building_id
        self.b5g = b5g
        self.use_gnn = use_gnn
        self.seed_val = seed

        self.current_step = 0

        # Obtener tensor de ejemplo para dimensionar observation_space
        if self.use_gnn:
            _, graph_tensor = graphs_to_tensor_synthetic(
                num_links=self.num_links,
                num_features=1,
                b5g=self.b5g,
                building_id=self.building_id
            )
            self.graph_tensor = graph_tensor
            obs_example = self.graph_tensor.flatten().float().numpy()
        else:
            _, mlp_tensor = graphs_to_tensor_synthetic(
                num_links=self.num_links,
                num_features=1,
                b5g=self.b5g,
                building_id=self.building_id
            )
            self.mlp_obs = mlp_tensor
            obs_example = self.mlp_obs.flatten().float().numpy()

        # Definir observation_space dinámicamente según el vector de ejemplo
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_example.shape,
            dtype=np.float32
        )

        # Action space: potencia por link y canal
        self.action_space = spaces.Box(
            low=0.0,
            high=self.p0,
            shape=(self.num_links, self.num_channels),
            dtype=np.float32
        )

        self.reset(seed=self.seed_val)

    def _get_obs(self):
        """Devuelve el vector plano de observación"""
        if self.use_gnn:
            return self.graph_tensor.flatten().float().numpy()
        else:
            return self.mlp_obs.flatten().float().numpy()

    def step(self, action: np.ndarray):
        self.current_step += 1

        # Aplicar constraints de potencia
        action = power_constraint(action, self.p0)

        # Calcular tasas por link y canal
        rates = nuevo_get_rates(
            phi=action,
            channel_matrix_batch=self.graph_tensor if self.use_gnn else self.mlp_obs,
            sigma=self.sigma,
            p0=self.p0
        )

        reward = rates.sum() - self.lambda_power * np.sum(action ** 2)

        done = self.current_step >= self.max_steps

        info: Dict[str, Any] = {
            "rates": rates,
            "total_power": np.sum(action)
        }

        return self._get_obs(), reward, done, False, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Regenerar graph/MLP obs
        if self.use_gnn:
            _, self.graph_tensor = graphs_to_tensor_synthetic(
                num_links=self.num_links,
                num_features=1,
                b5g=self.b5g,
                building_id=self.building_id
            )
        else:
            _, self.mlp_obs = graphs_to_tensor_synthetic(
                num_links=self.num_links,
                num_features=1,
                b5g=self.b5g,
                building_id=self.building_id
            )

        return self._get_obs(), {}

