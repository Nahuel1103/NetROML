"""
Entorno de Optimización de Redes Wi-Fi para Gymnasium
Implementa la simulación de una red multi-enlace con restricciones de canal ypotencia.
"""

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np

from utils import (
    objective_function,
    power_constraint_per_ap,
    mu_update_per_ap,
    nuevo_get_rates
)


class NetworkEnvironment(gym.Env):
    """
    Entorno para optimización de potencia en redes Wi-Fi.
    
    Gestiona la asignación de potencia across múltiples enlaces y canales,
    respetando restricciones físicas y maximizando la capacidad de la red.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, 
                channel_matrix_tensor ,  
                building_id: int = 990, 
                b5g: int = 0, 
                num_links: int = 5,
                num_channels: int = 3,
                num_power_levels: int = 2,
                num_layers: int = 5,
                K: int = 3,
                max_steps: int = 100, 
                eps: float = 5e-4,
                mu_lr: float = 1e-4, 
                synthetic: int = 0, 
                max_antenna_power_dbm: int = 6,
                sigma: float = 1e-4):
        super().__init__()

        # data externo (pasado por train.py)
        self.data = data
        self.data_size = len(data)

        # Parámetros de configuración del sistema
        self.building_id = building_id
        self.b5g = b5g
        self.num_links = num_links
        self.num_channels = num_channels
        self.num_power_levels = num_power_levels
        self.num_layers = num_layers
        self.K = K
        self.eps = eps
        self.mu_lr = mu_lr
        self.synthetic = synthetic
        self.max_antenna_power_dbm = max_antenna_power_dbm
        self.sigma = sigma

        # Configuración de episodios
        self.max_steps = max_steps
        self.step_count = 0
        self.current_data_idx = 0

        # Estado interno del entorno
        self.current_channel_matrix = None
        self.current_graph_data = None
        self.mu_k = None
        self.pmax_per_ap = None

        # Inicialización de componentes
        self._load_data()
        self.data_iterator = iter(self.dataloader)
        self._initialize_system_parameters()

        # Configuración de espacios Gymnasium
        self.num_actions = 1 + num_channels * num_power_levels
        self.action_space = spaces.MultiDiscrete([(self.num_actions)] * self.num_links)
        
        self.observation_space = spaces.Box(
            low=np.array([[0, 0]] * self.num_links, dtype=np.float32), 
            high=np.array([[self.num_channels, self.num_power_levels]] * self.p0, dtype=np.float32), 
            dtype=np.float32
        )

    def _initialize_system_parameters(self):
        """Configura parámetros físicos del sistema de comunicaciones."""
        self.p0 = 10 ** (self.max_antenna_power_dbm / 10)
        self.pmax_per_ap = 0.8 * self.p0 * torch.ones((self.num_links,))
        self.power_levels = torch.tensor(
            (np.arange(1, self.num_power_levels + 1) / self.num_power_levels) * self.p0
        )

    def reset(self, seed=None, options=None):
        """Inicializa un nuevo episodio de entrenamiento."""
        super().reset(seed=seed)

        # Obtener siguiente muestra del data
        self.current_graph_data = self.data[self.current_data_idx]
        self.current_data_idx = (self.current_data_idx + 1) % self.data_size
        
        # Procesamiento de matriz de canal
        # El data contiene matrices individuales (batch_size=1 implícito)
        channel_matrix = self.current_graph_data.matrix
        if channel_matrix.dim() == 2:
            self.current_channel_matrix = channel_matrix
        else:
            # Si viene con batch dimension, extraer primer elemento
            self.current_channel_matrix = channel_matrix.view(self.num_links, self.num_links)
        
        self.mu_k = torch.ones((self.num_links,), requires_grad=False)
        self.step_count = 0

        observation = self._get_observation()
        info = {
            "graph_data": self.current_graph_data,
            "mu_k": self.mu_k.numpy()
        }

        return observation, info

    def step(self, action):
        """Ejecuta un paso de simulación en el entorno."""
        phi = self._actions_to_phi(action)
        
        power_constr = power_constraint_per_ap(phi, self.pmax_per_ap)
        power_constr_mean = torch.mean(power_constr, dim=(0,1))
        
        self.mu_k = mu_update_per_ap(self.mu_k, power_constr, self.eps)
        
        reward = self._calculate_reward(phi, power_constr)
        
        # Transición al siguiente estado
        self.current_graph_data = self.data[self.current_data_idx]
        self.current_data_idx = (self.current_data_idx + 1) % self.data_size
        
        channel_matrix = self.current_graph_data.matrix
        if channel_matrix.dim() == 2:
            self.current_channel_matrix = channel_matrix
        else:
            self.current_channel_matrix = channel_matrix.view(self.num_links, self.num_links)

        observation = self._get_observation()

        info = {
            "power_allocation": phi.numpy(),
            "sum_rate": reward,
            "step": self.step_count,
            "mu_k": self.mu_k.numpy(),
            "power_constraint": power_constr_mean.item(),
            "graph_data": self.current_graph_data,
            "cost": reward
        }

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Construye el vector de observación para el agente."""
        channel_matrix_np = self.current_channel_matrix.numpy()
        mu_k_np = self.mu_k.numpy().reshape(-1, 1)
        observation = np.concatenate([channel_matrix_np, mu_k_np], axis=1)
        return observation.astype(np.float32)

    def _get_info(self, phi=None, reward=None, power_constr=None):
        """
        Info con TODA la información del grafo.
        Aquí es donde la política GNN obtendrá los datos estructurados.
        """
        info = {
            # CRÍTICO: Datos del grafo completo para GNN
            'graph_data': self.current_graph_data,
            'channel_matrix': self.current_channel_matrix.numpy(),
            'mu_k': self.mu_k.numpy().copy(),
            'step': self.step_count,
            'num_links': self.num_links,
        }
        
        if phi is not None:
            info["power_allocation"] = phi.numpy()
            info["sum_rate"] = reward
            info["power_constraint"] = power_constr.item() if torch.is_tensor(power_constr) else power_constr
        
        return info    

    def _actions_to_phi(self, actions):
        """
        Convierte acciones discretas en matriz de asignación de potencia.
        
        Mapeo de acciones:
        - Acción 0: No transmitir
        - Acciones 1-6: Combinaciones de (canal, nivel_potencia)
          - action = 1 + channel_idx * num_power_levels + power_idx
        """
        actions_tensor = torch.from_numpy(actions).long()
        phi = torch.zeros(1, self.num_links, self.num_channels)
        
        for link_idx in range(self.num_links):
            action = actions_tensor[link_idx].item()
            
            if action > 0:
                action_offset = action - 1
                channel_idx = action_offset // self.num_power_levels
                power_idx = action_offset % self.num_power_levels
                
                if channel_idx < self.num_channels and power_idx < self.num_power_levels:
                    phi[0, link_idx, channel_idx] = self.power_levels[power_idx]
        
        return phi

    def _calculate_reward(self, phi, power_constr):
        """Calcula la recompensa basada en métricas de la red."""
        rates = nuevo_get_rates(phi, self.current_channel_matrix.unsqueeze(0), self.sigma)
        
        if not torch.isfinite(rates).all():
            return -100.0
        
        sum_rate = -objective_function(rates)
        
        if not torch.isfinite(sum_rate):
            return -100.0
        
        penalty = (power_constr * self.mu_k.unsqueeze(0)).sum(dim=1)
        
        if not torch.isfinite(penalty).all():
            penalty = torch.zeros_like(penalty)
        
        reward = (sum_rate - penalty).mean().item()
        
        if not np.isfinite(reward):
            return -100.0
        
        return reward

    def render(self):
        """Proporciona visualización del estado actual."""
        print(f"Paso: {self.step_count}")
        print(f"Mu_k: {self.mu_k.numpy()}")
        
    def close(self):
        """Libera recursos del entorno."""
        pass