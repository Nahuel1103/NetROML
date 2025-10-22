"""
Entorno de Optimización de Redes Wi-Fi para Gymnasium
Implementa la simulación de una red multi-enlace con restricciones de canal y potencia.
"""

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np

from utils import objective_function


class NetworkEnvironment(gym.Env):
    """
    Entorno para optimización de potencia en redes Wi-Fi.
    
    Gestiona la asignación de canales y potencia entre múltiples enlaces,
    respetando restricciones físicas y maximizando el rate de cada AP.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 num_links: int = 5,
                 num_channels: int = 3,
                 num_power_levels: int = 2,
                 max_steps: int = 100,
                 eps: float = 5e-4,
                 max_antenna_power_dbm: int = 6,
                 sigma: float = 1e-4,
                 device: str = "cpu",
                 channel_matrix_iter = None):
        super().__init__()

        # Parámetros de configuración del sistema
        self.num_links = num_links
        self.num_channels = num_channels
        self.num_power_levels = num_power_levels
        self.eps = eps
        self.max_antenna_power_dbm = max_antenna_power_dbm
        self.sigma = sigma
        self.device = torch.device(device)

        # Channel matrix
        self.channel_matrix = channel_matrix_iter
        self.channel_matrix_iter = None

        # Initialize the state
        self.step_count = 0
        self.state = None
        self.rate = None

        # Inicializo parámetros internos
        self._initialize_system_parameters()

        # Action space: 0 = no transmitir, 1..C*P = elegir canal+potencia
        self.num_action_per_link =  1 + self.num_channels * self.num_power_levels
        self.action_space = spaces.MultiDiscrete([self.num_action_per_link] * self.num_links)

        # Observation space
        self.observation_space = spaces.Dict({
            'channel_matrix': spaces.Box(
                low=-100.0,
                high=100.0,
                shape=(self.num_links, self.num_links),
                dtype=np.float32
            ),
            'mu_k': spaces.Box(
                low=0.0,
                high=1e3,
                shape=(self.num_links,),
                dtype=np.float32
            )
        })

    def _initialize_system_parameters(self):
        """Configuración de algunos parámetros del sistema"""
        # p0 en Watts
        self.p0 = 10 ** (self.max_antenna_power_dbm / 10.0)
        # Potencia máxima por AP
        self.pmax_per_ap = (0.95 * self.p0) * torch.ones((self.num_links,), device=self.device, dtype=torch.float32)
        # Niveles discretos de potencia
        self.power_levels = torch.linspace(self.p0 / self.num_power_levels,
                                           self.p0,
                                           self.num_power_levels,
                                           device=self.device,
                                           dtype=torch.float32)

    def reset(self, seed = None, options = None):
        """Inicializa nuevo episodio."""

        super().reset(seed=seed)

        if self.channel_matrix is not None:
            self.channel_matrix_iter = iter(self.channel_matrix)

        # Inicializo la primera matriz de canal del episodio
        self.channel_matrix = self._get_channel_matrix()


        # Variable dual por AP
        self.mu_k = torch.ones((self.num_links,), device=self.device, dtype=torch.float32)
        self.step_count = 0

        obs= self._get_observation()
        info = {"episode_start": True}

        return obs, info
    
    def step(self, action):
        """Ejecuta una acción."""

        # Obtengo la matriz de canal
        self.channel_matrix = self._get_channel_matrix()

        # Convierto una acción a phi
        self.phi = self._actions_to_phi(action)

        # 
        self.power_constr = self._calculate_power_constraint()

        self.mu_k = self._mu_update_per_ap() 

        self.rate = self._get_rates()

        reward = self._calculate_reward()

        obs = self._get_observation()
        
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "power_constraint": float(torch.mean(self.power_constr).item()),
            "rate": self.rate
        }

        return obs, reward, terminated, truncated, info

    def _get_channel_matrix(self):

        self.channel_matrix = next(self.channel_matrix_iter)

        return self.channel_matrix
    
    def _actions_to_phi(self, action):
        """Convierte acciones en matriz de asignación de canal y potencia"""

        phi = np.zeros((self.num_links, self.num_channels))

        for link_idx in range(self.num_links):
            act = action[link_idx]
            if act == 0:
                continue
            act -= 1
            ch_idx = act // self.num_power_levels
            pw_idx = act % self.num_power_levels
            if ch_idx < self.num_channels:
                phi[link_idx, ch_idx] = self.power_levels[pw_idx].item()
        
        phi = torch.tensor(phi, dtype=torch.float32)

        return phi
    
    # def actions_to_phi_torch(actions, num_links, num_channels, num_power_levels, power_levels, device="cpu"):
    #     # actions: torch tensor shape [num_links], dtype long/int
    #     # power_levels: torch tensor shape [num_power_levels]
    #     actions = actions.clone().long().to(device)
    #     phi = torch.zeros((num_links, num_channels), dtype=torch.float32, device=device)

    #     mask_tx = actions != 0  # enlaces que transmiten
    #     if mask_tx.sum() == 0:
    #         return phi

    #     acts = actions[mask_tx] - 1  # 0..C*P-1
    #     ch_idx = (acts // num_power_levels).long()
    #     pw_idx = (acts % num_power_levels).long()

    #     # evitar índices fuera de rango
    #     valid = ch_idx < num_channels
    #     if valid.sum() == 0:
    #         return phi

    #     # indices globales de links que transmiten y son válidos
    #     links_idx = torch.nonzero(mask_tx).squeeze(1)[valid]
    #     ch_idx = ch_idx[valid]
    #     pw_idx = pw_idx[valid]

    #     phi[links_idx, ch_idx] = power_levels[pw_idx]

    #     return phi

    def _calculate_power_constraint(self):
        """
        Devuelve vector (power_per_link - pmax_per_ap), shape [num_links].

        Asume phi.shape == [num_links, num_channels] (2D) — cada fila es un enlace.
        """

        return torch.sum(self.phi, dim=1) - self.pmax_per_ap

    def _mu_update_per_ap(self):
        """Actualización de la variable dual"""

        mu_k = self.mu_k.detach()
        mu_k_update = self.eps * self.power_constr  
        mu_k = mu_k + mu_k_update
        mu_k = torch.max(mu_k, torch.tensor(0.0))

        return mu_k

    def _get_rates(self):
        """"""

        num_links, num_channels = self.phi.shape

        # Señal útil
        diagH = torch.diagonal(self.channel_matrix)
        signal =  diagH.unsqueeze(-1) * self.phi

        # Máscara para eliminar diagonal y calcular interferencia
        mask = (1 - torch.eye(num_links, device=self.device))
        diagH_off = self.channel_matrix * mask

        # Interferencia por receptor y canal: diagH_off @ phi
        interference = diagH_off @ self.phi 

        # Calcular SNR por enlace y canal
        snr = signal / (self.sigma + interference)

        # Calcular rate total por enlace sumando sobre canales
        rates = torch.log1p(torch.sum(snr, dim=-1))  

        return rates

    def _objective_function(self):
        """"""

        sum_rate = torch.sum(self.rate)  
        return sum_rate    

    def _calculate_reward(self):

        if not torch.isfinite(self.rate).all():
            return -100.0

        sum_rate = -self._objective_function()
        if not torch.isfinite(sum_rate):
            return -100.0

        penalty = (self.power_constr * self.mu_k).sum()
        if not torch.isfinite(penalty).all():
            penalty = torch.zeros_like(penalty)

        reward = (sum_rate - penalty).mean().item()
        return reward if np.isfinite(reward) else -100.0
    
    def _get_observation(self):
        """"""
        return {
            'channel_matrix': self.channel_matrix.numpy().astype(np.float32),
            'mu_k': self.mu_k.numpy().astype(np.float32)
        }


    def render(self):
        print(f"Step: {self.step_count}, Mu_k: {self.mu_k.numpy()}")
    
    def close(self):
        pass