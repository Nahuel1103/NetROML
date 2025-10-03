"""
Entorno de Optimización de Redes Wi-Fi para Gymnasium
Implementa la simulación de una red multi-enlace con restricciones de potencia.
"""

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from utils import (
    graphs_to_tensor,
    graphs_to_tensor_synthetic,
    get_gnn_inputs,
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

    def __init__(self, 
                 building_id: int = 990, 
                 b5g: int = 0, 
                 num_links: int = 5,
                 num_channels: int = 3,
                 num_power_levels: int = 2,
                 num_layers: int = 5,
                 K: int = 3,
                 batch_size: int = 64, 
                 epochs: int = 100, 
                 eps: float = 5e-4,
                 mu_lr: float = 1e-4, 
                 synthetic: int = 0, 
                 max_antenna_power_dbm: int = 6,
                 sigma: float = 1e-4):
        super().__init__()

        # Parámetros de configuración del sistema
        self.building_id = building_id
        self.b5g = b5g
        self.num_links = num_links
        self.num_channels = num_channels
        self.num_power_levels = num_power_levels
        self.num_layers = num_layers
        self.K = K
        self.batch_size = batch_size
        self.epochs = epochs
        self.eps = eps
        self.mu_lr = mu_lr
        self.synthetic = synthetic
        self.max_antenna_power_dbm = max_antenna_power_dbm
        self.sigma = sigma

        # Configuración de episodios
        self.max_steps = epochs
        self.step_count = 0

        # Estado interno del entorno
        self.current_channel_matrix = None
        self.current_graph_data = None
        self.mu_k = None
        self.pmax_per_ap = None

        # Inicialización de componentes
        self._load_dataset()
        self.data_iterator = iter(self.dataloader)
        self._initialize_system_parameters()

        # Configuración de espacios Gymnasium
        self.num_actions = 1 + num_channels * num_power_levels
        self.action_space = spaces.MultiDiscrete([self.num_actions] * self.num_links)
        
        obs_shape = (num_links, num_links + 1)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=np.inf, 
            shape=obs_shape,
            dtype=np.float32
        )

    def _initialize_system_parameters(self):
        """Configura parámetros físicos del sistema de comunicaciones."""
        self.p0 = 10 ** (self.max_antenna_power_dbm / 10)
        self.pmax_per_ap = 0.8 * self.p0 * torch.ones((self.num_links,))
        self.power_levels = torch.tensor((np.arange(1, self.num_power_levels + 1) / self.num_power_levels) * self.p0)

    def _load_dataset(self):
        """Carga dataset de matrices de canal para entrenamiento."""
        if self.synthetic:
            x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
                num_links=self.num_links,
                num_features=1,
                b5g=self.b5g,
                building_id=self.building_id
            )
            dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
            self.dataset = dataset[:7000]
        else:
            x_tensor, channel_matrix_tensor = graphs_to_tensor(
                train=True,
                num_links=self.num_links,
                num_features=1,
                b5g=self.b5g,
                building_id=self.building_id
            )
            dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
            self.dataset = dataset
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

    def reset(self, seed=None, options=None):
        """Inicializa un nuevo episodio de entrenamiento."""
        super().reset(seed=seed)

        try:
            self.current_graph_data = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            self.current_graph_data = next(self.data_iterator)
        
        # Procesamiento de matriz de canal
        channel_matrix_batch = self.current_graph_data.matrix
        channel_matrix_batch = channel_matrix_batch.view(self.batch_size, self.num_links, self.num_links)
        self.current_channel_matrix = channel_matrix_batch[0]
        
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
        try:
            self.current_graph_data = next(self.data_iterator)
            channel_matrix_batch = self.current_graph_data.matrix
            channel_matrix_batch = channel_matrix_batch.view(self.batch_size, self.num_links, self.num_links)
            self.current_channel_matrix = channel_matrix_batch[0]
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            self.current_graph_data = next(self.data_iterator)
            channel_matrix_batch = self.current_graph_data.matrix
            channel_matrix_batch = channel_matrix_batch.view(self.batch_size, self.num_links, self.num_links)
            self.current_channel_matrix = channel_matrix_batch[0]

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
        
        # Iterar sobre cada enlace
        for link_idx in range(self.num_links):
            action = actions_tensor[link_idx].item()
            
            # Acción 0 = no transmitir (phi permanece en 0)
            if action > 0:
                action_offset = action - 1  # Convertir a índice base-0
                
                # Decodificar: action_offset = channel_idx * num_power_levels + power_idx
                channel_idx = action_offset // self.num_power_levels
                power_idx = action_offset % self.num_power_levels
                
                # Validación de índices
                if channel_idx < self.num_channels and power_idx < self.num_power_levels:
                    phi[0, link_idx, channel_idx] = self.power_levels[power_idx]
                else:
                    print(f"WARNING: Índice inválido en link {link_idx}: "
                          f"action={action}, channel={channel_idx}, power={power_idx}")
        
        return phi

    def _calculate_reward(self, phi, power_constr):
        """
        Calcula la recompensa basada en métricas de la red.
        FIX: Manejo robusto de NaN/Inf
        """
        rates = nuevo_get_rates(phi, self.current_channel_matrix.unsqueeze(0), self.sigma)
        
        # Protección contra NaN/Inf
        if not torch.isfinite(rates).all():
            print("WARNING: Rates contiene NaN/Inf, usando valor por defecto")
            return -100.0
        
        sum_rate = -objective_function(rates)
        
        if not torch.isfinite(sum_rate):
            print("WARNING: sum_rate es NaN/Inf")
            return -100.0
        
        penalty = (power_constr * self.mu_k.unsqueeze(0)).sum(dim=1)
        
        if not torch.isfinite(penalty).all():
            print("WARNING: penalty contiene NaN/Inf")
            penalty = torch.zeros_like(penalty)
        
        reward = (sum_rate - penalty).mean().item()
        
        # Protección final
        if not np.isfinite(reward):
            print("WARNING: reward final es NaN/Inf")
            return -100.0
        
        return reward

    def render(self):
        """Proporciona visualización del estado actual."""
        print(f"Paso: {self.step_count}")
        print(f"Mu_k: {self.mu_k.numpy()}")
        
    def close(self):
        """Libera recursos del entorno."""
        pass