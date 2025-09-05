import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple

from utils import graphs_to_tensor, graphs_to_tensor_synthetic, power_constraint_per_ap, nuevo_get_rates, objective_function

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
        seed: int = 12345,
        discrete_actions: bool = True,
        num_power_levels: int = 2
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
        self.discrete_actions = discrete_actions
        self.num_power_levels = num_power_levels

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

        # Action space
        if self.discrete_actions:
            # Esquema actualizado: power_levels incluye 0 directamente
            # num_actions = num_channels * num_power_levels (sin +1 adicional)
            self.num_actions = self.num_channels * self.num_power_levels
            self.action_space = spaces.MultiDiscrete([self.num_actions] * self.num_links)
            # definir niveles de potencia siguiendo train.py: [0, p0/2, p0]
            if self.num_power_levels == 1:
                self.power_levels = torch.tensor([self.p0])
            elif self.num_power_levels == 2:
                self.power_levels = torch.tensor([self.p0 / 2, self.p0])
            elif self.num_power_levels == 3:
                self.power_levels = torch.tensor([0, self.p0 / 2, self.p0])
            else:
                # construir niveles equiespaciados desde 0 hasta p0
                self.power_levels = torch.linspace(0, self.p0, self.num_power_levels)
        else:
            # acción continua: potencia por link y canal
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

    def _decode_discrete_actions(self, action: np.ndarray) -> torch.Tensor:
        """
        Convierte acciones discretas por enlace en un tensor phi [1, num_links, num_channels]
        siguiendo el esquema actualizado de train.py: power_levels incluye 0 directamente.
        """
        batch_size = 1
        phi = torch.zeros(batch_size, self.num_links, self.num_channels, dtype=torch.float32)
        if action.ndim == 0:
            action = np.array([int(action)])
        for link_idx in range(self.num_links):
            a = int(action[link_idx]) if action.shape[0] > 1 else int(action.item())
            channel_idx = a // self.num_power_levels
            power_idx = a % self.num_power_levels
            power_val = self.power_levels[power_idx].item()
            if 0 <= channel_idx < self.num_channels:
                phi[0, link_idx, channel_idx] = power_val
        return phi

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # Construir phi según tipo de acción
        if self.discrete_actions:
            phi = self._decode_discrete_actions(action)
        else:
            # acción continua -> asegurar rango y convertir a torch con batch dim
            a = np.clip(action, 0.0, self.p0).astype(np.float32)
            phi = torch.from_numpy(a).unsqueeze(0)

        # Calcular tasas por link y canal
        channel_tensor = self.graph_tensor if self.use_gnn else self.mlp_obs
        if channel_tensor.dim() == 2:
            channel_tensor = channel_tensor.unsqueeze(0)

        rates = nuevo_get_rates(
            phi=phi,
            channel_matrix_batch=channel_tensor,
            sigma=self.sigma,
            p0=self.p0
        )

        # Objetivo (sum-rate) y penalización por potencia por-AP
        sum_rate = objective_function(rates).mean().item()
        pmax_per_ap = torch.full((self.num_links,), self.p0 / 2, dtype=torch.float32)
        power_excess_per_ap = power_constraint_per_ap(phi, pmax_per_ap)
        # solo exceso positivo
        power_excess = torch.clamp(power_excess_per_ap, min=0.0).mean().item()

        reward = sum_rate - self.lambda_power * power_excess

        done = self.current_step >= self.max_steps

        info: Dict[str, Any] = {
            "sum_rate": sum_rate,
            "power_excess": power_excess
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

