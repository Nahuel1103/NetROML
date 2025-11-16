"""
Entorno de Optimización de Redes Wi-Fi para Gymnasium
Implementa la simulación de una red multi-enlace con restricciones de canal y potencia.
"""

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np


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
                 channel_matrix_iter=None):
        super().__init__()

        # Parámetros de configuración del sistema
        self.num_links = num_links
        self.num_channels = num_channels
        self.num_power_levels = num_power_levels
        self.max_steps = max_steps
        self.eps = eps
        self.max_antenna_power_dbm = max_antenna_power_dbm
        self.sigma = sigma
        self.device = torch.device(device)

        # Guardar el iterador
        self.channel_matrix_source = channel_matrix_iter
        self.channel_matrix_iter = None
        
        # Estado interno
        self.step_count = 0
        self.channel_matrix = None
        self.phi = None
        self.rate = None
        self.mu_k = None
        self.power_constr = None
        self.iterator_exhausted = False

        # Inicializar parámetros del sistema
        self._initialize_system_parameters()

        # Action space: 0 = no transmitir, 1..C*P = elegir canal+potencia
        self.num_action_per_link = 1 + self.num_channels * self.num_power_levels
        self.action_space = spaces.MultiDiscrete([self.num_action_per_link] * self.num_links)

        # Observation space
        self.observation_space = spaces.Dict({
            'channel_matrix': spaces.Box(
                low=-1.0,  
                high=100.0,
                shape=(self.num_links, self.num_links),
                dtype=np.float32
            ),
            'mu_k': spaces.Box(
                low=0.0,
                high=1.0,  
                shape=(self.num_links,),
                dtype=np.float32
            )
        })

    def reset(self, seed=None, options=None):
        """Inicializa nuevo episodio."""
        super().reset(seed=seed)

        # Reinicializar iterador
        if self.channel_matrix_source is not None:
            self.channel_matrix_iter = iter(self.channel_matrix_source)
        
        self.iterator_exhausted = False
        self.step_count = 0

        # Inicializar primera matriz de canal
        self.channel_matrix = self._get_channel_matrix()
        
        # Variable dual por AP
        self.mu_k = torch.zeros((self.num_links,), device=self.device, dtype=torch.float32)
        
        # Inicializar phi y rate
        self.phi = torch.zeros((self.num_links, self.num_channels), device=self.device, dtype=torch.float32)
        self.rate = torch.zeros((self.num_links,), device=self.device, dtype=torch.float32)
        self.power_constr = torch.zeros((self.num_links,), device=self.device, dtype=torch.float32)

        obs = self._get_observation()
        info = {"episode_start": True}

        return obs, info
    
    def step(self, action):
        """Ejecuta una acción."""
        self.step_count += 1

        # Verificar si el iterador se agotó
        if self.iterator_exhausted:
            obs = {
                'channel_matrix': -np.ones((self.num_links, self.num_links), dtype=np.float32),
                'mu_k': self.mu_k.cpu().numpy()
            }
            return obs, 0.0, True, False, {"reason": "iterator_exhausted"}

        # Obtener nueva matriz de canal
        self.channel_matrix = self._get_channel_matrix()
        
        # Verificar si el iterador se agotó después de obtener matriz
        if self.iterator_exhausted:
            obs = {
                'channel_matrix': -np.ones((self.num_links, self.num_links), dtype=np.float32),
                'mu_k': self.mu_k.cpu().numpy()
            }
            return obs, 0.0, True, False, {"reason": "iterator_exhausted"}

        # Convertir acción a phi
        self.phi = self._actions_to_phi(action)

        # Calcular restricción de potencia
        self.power_constr = self._calculate_power_constraint()

        # Actualizar multiplicadores de Lagrange
        self.mu_k = self._mu_update_per_ap()

        # Calcular rates
        self.rate = self._get_rates()

        # Calcular recompensa
        reward = self._calculate_reward()

        # Obtener observación
        obs = self._get_observation()
        
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "power_constraint": float(torch.mean(self.power_constr).item()),
            "rate": float(torch.sum(self.rate).item()),
            "avg_mu": float(torch.mean(self.mu_k).item())
        }

        return obs, reward, terminated, truncated, info

    def _initialize_system_parameters(self):
        """Configuración de parámetros del sistema"""
        # p0 en Watts
        self.p0 = 10 ** (self.max_antenna_power_dbm / 10.0)
        
        # Potencia máxima por AP
        self.pmax_per_ap = (0.95 * self.p0) * torch.ones(
            (self.num_links,), device=self.device, dtype=torch.float32
        )
        
        # Niveles discretos de potencia
        self.power_levels = torch.linspace(
            self.p0 / self.num_power_levels,
            self.p0,
            self.num_power_levels,
            device=self.device,
            dtype=torch.float32
        )

    def _get_channel_matrix(self):
        """Obtiene siguiente matriz de canal del iterador"""
        if self.channel_matrix_iter is None:
            raise ValueError("No hay iterador de matrices de canal definido.")
        
        try:
            matrix = next(self.channel_matrix_iter)
            
            # Convertir a tensor de torch si viene como numpy
            if isinstance(matrix, np.ndarray):
                matrix = torch.from_numpy(matrix.astype(np.float32))
            
            # Mover al device correcto
            if not isinstance(matrix, torch.Tensor):
                matrix = torch.tensor(matrix, dtype=torch.float32)
            
            matrix = matrix.to(self.device)
            
            return matrix
            
        except StopIteration:
            # Señalizar fin del iterador
            self.iterator_exhausted = True
            return -torch.ones((self.num_links, self.num_links), 
                              device=self.device, dtype=torch.float32)
    
    def _actions_to_phi(self, action):
        """
        Convierte acciones en matriz de asignación de canal y potencia
        
        action: array de shape [num_links]
        Returns: tensor de shape [num_links, num_channels]
        """
        phi = torch.zeros((self.num_links, self.num_channels), 
                         dtype=torch.float32, device=self.device)

        for link_idx in range(self.num_links):
            act = int(action[link_idx])
            
            if act == 0:  # No transmitir
                continue
            
            act -= 1  # Ajustar índice (0 era "no transmitir")
            ch_idx = act // self.num_power_levels
            pw_idx = act % self.num_power_levels
            
            if ch_idx < self.num_channels:
                phi[link_idx, ch_idx] = self.power_levels[pw_idx]
        
        return phi

    def _calculate_power_constraint(self):
        """
        Calcula la violación de restricción de potencia por enlace
        
        Returns: tensor [num_links] con (power_used - pmax) por enlace
        """
        power_per_link = torch.sum(self.phi, dim=1)  # [num_links]
        return power_per_link - self.pmax_per_ap

    def _mu_update_per_ap(self):
        """
        Actualización de variable dual usando método del gradiente proyectado
        
        Returns: tensor [num_links] con multiplicadores actualizados
        """
        mu_k_new = self.mu_k + self.eps * self.power_constr
        mu_k_new = torch.clamp(mu_k_new, min=0.0)  # Proyección al dominio válido
        return mu_k_new

    def _get_rates(self):
        """
        Calcula tasa de transmisión por enlace usando capacidad de Shannon
        
        Returns: tensor [num_links] con rate por enlace
        """
        num_links, num_channels = self.phi.shape

        # Señal útil: diagonal de H multiplicada por potencia transmitida
        diagH = torch.diagonal(self.channel_matrix, dim1=0, dim2=1)  # [num_links]
        signal = diagH.unsqueeze(-1) * self.phi  # [num_links, num_channels]

        # Interferencia: señal recibida de otros transmisores
        # Máscara para eliminar diagonal
        mask = 1 - torch.eye(num_links, device=self.device, dtype=torch.float32)
        H_off_diag = self.channel_matrix * mask  # [num_links, num_links]
        
        # Interferencia total por receptor y canal
        interference = torch.matmul(H_off_diag, self.phi)  # [num_links, num_channels]

        # SINR por enlace y canal
        eps_safe = 1e-12
        snr = signal / (self.sigma + interference + eps_safe)

        # Rate total por enlace (suma sobre canales)
        rates = torch.sum(torch.log1p(snr), dim=-1)  # [num_links]

        return rates

    def _calculate_reward(self):
        """
        Calcula recompensa como sum_rate - penalización por violar restricciones
        
        Returns: float con la recompensa
        """
        # Verificar validez de rates
        if not torch.isfinite(self.rate).all():
            return -100.0

        # Objetivo: maximizar sum rate (negativo para minimización)
        sum_rate = torch.sum(self.rate)
        
        if not torch.isfinite(sum_rate):
            return -100.0

        # Penalización por violación de restricción de potencia
        # Solo penalizar si power_constr > 0 (violación)
        penalty = torch.sum(torch.clamp(self.power_constr, min=0.0) * self.mu_k)
        
        if not torch.isfinite(penalty):
            penalty = torch.tensor(0.0, device=self.device)

        # Recompensa = sum_rate - penalty
        reward = (sum_rate - penalty).item()
        
        return reward if np.isfinite(reward) else -100.0
    
    def _get_observation(self):
        """
        Construye observación para el agente
        
        Returns: dict con 'channel_matrix' y 'mu_k'
        """
        return {
            'channel_matrix': self.channel_matrix.cpu().numpy().astype(np.float32),
            'mu_k': self.mu_k.cpu().numpy().astype(np.float32)
        }

    def render(self):
        """Muestra estado actual del entorno"""
        if self.phi is not None:
            print(f"\n{'='*60}")
            print(f"Step: {self.step_count}")
            print(f"Mu_k: {self.mu_k.cpu().numpy()}")
            print(f"Rates: {self.rate.cpu().numpy()}")
            print(f"Total Rate: {torch.sum(self.rate).item():.4f}")
            print(f"Power Constraint: {self.power_constr.cpu().numpy()}")
            print(f"{'='*60}\n")
    
    def close(self):
        """Cierra el entorno"""
        pass