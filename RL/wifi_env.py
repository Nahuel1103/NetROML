import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import random

# Importar tus módulos existentes
from gnn import GNN
from utils import (
    graphs_to_tensor_synthetic, 
    graphs_to_tensor,
    get_gnn_inputs,
    nuevo_get_rates,
    power_constraint_per_ap,
    objective_function,
    mu_update_per_ap
)

@dataclass
class WirelessConfig:
    """Configuración del entorno wireless usando tu estructura existente"""
    building_id: int = 990
    b5g: bool = False
    num_links: int = 5
    num_channels: int = 3
    num_layers: int = 5
    K: int = 3
    p0: float = 4.0  # Potencia máxima
    sigma: float = 1e-4  # Ruido
    synthetic: bool = True
    eps: float = 5e-4  # Para mu_update
    exploration_bonus: float = 0.01

class WirelessGNNEnvironment(gym.Env):
    """
    Entorno Gymnasium que usa tu GNN y funciones existentes
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, config: WirelessConfig = None):
        super().__init__()
        
        self.config = config or WirelessConfig()
        self.num_links = self.config.num_links
        self.num_channels = self.config.num_channels
        self.p0 = self.config.p0
        
        # Configuración de potencias (usando tu estructura existente)
        self.power_levels = torch.tensor([self.p0/2, self.p0])
        self.num_power_levels = len(self.power_levels)
        self.pmax_per_ap = 0.8*self.p0*torch.ones(self.num_links)
        
        # Cargar datos (usando tus funciones existentes)
        self._load_data()
        
        # Inicializar GNN (usando tu arquitectura existente)
        self._setup_gnn()
        
        # Definir espacios
        self._setup_spaces()
        
        # Estado del entorno
        self.current_data_idx = 0
        self.current_phi = None
        self.mu_k = torch.ones(self.num_links, requires_grad=False)
        self.step_count = 0
        
        # Tracking de exploración
        self.exploration_stats = self._init_exploration_tracking()
        
    def _load_data(self):
        """Cargar datos usando tus funciones existentes"""
        if self.config.synthetic:
            x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
                num_links=self.num_links,
                num_features=1,
                b5g=self.config.b5g,
                building_id=self.config.building_id
            )
        else:
            x_tensor, channel_matrix_tensor = graphs_to_tensor(
                train=True,
                num_links=self.num_links,
                num_features=1,
                b5g=self.config.b5g,
                building_id=self.config.building_id
            )
        
        # Convertir a dataset GNN usando tu función
        self.dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        self.dataset_size = len(self.dataset)
        
        print(f"Datos cargados: {self.dataset_size} muestras")
        
    def _setup_gnn(self):
        """Configurar GNN usando tu arquitectura existente"""
        input_dim = 1
        hidden_dim = 1
        
        # Calcular dimensiones como en tu código
        num_actions = 1 + self.num_channels*self.num_power_levels
        output_dim = num_actions
        
        self.gnn_model = GNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=self.config.num_layers,
            dropout=False,
            K=self.config.K
        )
        
        print(f"GNN configurado: {input_dim} -> {hidden_dim} -> {output_dim}")
        print(f"Acciones posibles por link: {num_actions}")
        
    def _setup_spaces(self):
        """Configurar espacios de acción y observación"""
        # Espacio de acciones: cada link elige una acción
        num_actions = 1 + self.num_channels*self.num_power_levels
        self.action_space = spaces.MultiDiscrete([num_actions]*self.num_links)
        
        # Espacio de observación: usar las características del grafo
        # Por simplicidad, usar vector plano con info del grafo actual
        obs_dim = (
            self.num_links * self.num_links +  # Channel matrix flattened
            self.num_links +                   # Current power per link
            self.num_links                     # Exploration bonus per link
        )
        
        self.observation_space = spaces.Box(
            low=0, 
            high=self.p0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
    def _init_exploration_tracking(self):
        """Inicializar tracking de exploración por link"""
        stats = {}
        total_combinations = self.num_channels * self.num_power_levels
        
        for link_idx in range(self.num_links):
            stats[link_idx] = {
                'combinations_seen': set(),
                'channels_used': set(),
                'powers_used': set(),
                'total_combinations': total_combinations
            }
        return stats
    
    def _decode_actions_to_phi(self, actions: np.ndarray) -> torch.Tensor:
        """
        Convertir acciones del entorno a phi usando tu lógica existente
        """
        phi = torch.zeros(1, self.num_links, self.num_channels)  # batch_size=1
        
        for link_idx, action in enumerate(actions):
            if action > 0:  # Si no es "no transmitir"
                actions_active = action - 1
                channel_idx = actions_active // self.num_power_levels
                power_idx = actions_active % self.num_power_levels
                
                if channel_idx < self.num_channels and power_idx < self.num_power_levels:
                    phi[0, link_idx, channel_idx] = self.power_levels[power_idx]
                    
                    # Actualizar tracking
                    combination = (channel_idx, power_idx)
                    self.exploration_stats[link_idx]['combinations_seen'].add(combination)
                    self.exploration_stats[link_idx]['channels_used'].add(channel_idx)
                    self.exploration_stats[link_idx]['powers_used'].add(power_idx)
        
        return phi
    
    def _get_current_data(self):
        """Obtener el dato actual del dataset"""
        return self.dataset[self.current_data_idx]
    
    def _compute_gnn_probabilities(self, data):
        """Usar tu GNN para obtener probabilidades"""
        with torch.no_grad():
            # Expandir para batch_size=1
            data_batch = data.clone()
            data_batch.x = data_batch.x.unsqueeze(0) if data_batch.x.dim() == 2 else data_batch.x
            
            psi = self.gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            psi = psi.view(self.num_links, -1)  # [num_links, num_actions]
            
            probs = torch.softmax(psi, dim=-1)
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            return probs
    
    def _get_observation(self):
        """Construir observación del estado actual"""
        data = self._get_current_data()
        
        # Channel matrix flattened
        channel_matrix_flat = data.matrix.flatten()
        
        # Current power per link
        if self.current_phi is not None:
            current_power = self.current_phi.sum(dim=2).flatten()  # Sum over channels
        else:
            current_power = torch.zeros(self.num_links)
        
        # Exploration bonus per link
        exploration_bonus = torch.zeros(self.num_links)
        for link_idx in range(self.num_links):
            coverage = len(self.exploration_stats[link_idx]['combinations_seen']) / \
                      self.exploration_stats[link_idx]['total_combinations']
            exploration_bonus[link_idx] = coverage * self.config.exploration_bonus
        
        # Concatenar
        obs = torch.cat([
            channel_matrix_flat.float(),
            current_power.float(),
            exploration_bonus.float()
        ])
        
        return obs.numpy().astype(np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reiniciar entorno"""
        super().reset(seed=seed)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Elegir dato aleatorio del dataset
        self.current_data_idx = np.random.randint(0, self.dataset_size)
        self.current_phi = None
        self.step_count = 0
        
        # Reiniciar mu_k
        self.mu_k = torch.ones(self.num_links, requires_grad=False)
        
        # Reiniciar tracking
        self.exploration_stats = self._init_exploration_tracking()
        
        observation = self._get_observation()
        info = {
            "data_idx": self.current_data_idx,
            "step": self.step_count
        }
        
        return observation, info
    
    def step(self, actions: np.ndarray):
        """Ejecutar un step usando tu lógica existente"""
        # Obtener datos actuales
        data = self._get_current_data()
        
        # Convertir acciones a phi
        phi = self._decode_actions_to_phi(actions)
        self.current_phi = phi
        
        # Calcular métricas usando tus funciones existentes
        channel_matrix_batch = data.matrix.unsqueeze(0)  # Add batch dim
        
        # Rates usando tu función
        rates = nuevo_get_rates(phi, channel_matrix_batch, self.config.sigma, self.p0)
        
        # Power constraints usando tu función
        power_constr = power_constraint_per_ap(phi, self.pmax_per_ap)
        
        # Objective function usando tu función
        sum_rate = objective_function(rates)
        
        # Update mu_k usando tu función
        self.mu_k = mu_update_per_ap(self.mu_k, power_constr, self.config.eps)
        
        # Calcular reward
        sum_rate_value = sum_rate.item()
        power_penalty = torch.clamp(power_constr, min=0).sum().item() * 10.0
        
        # Exploration bonus
        exploration_bonus = 0
        for link_idx in range(self.num_links):
            coverage = len(self.exploration_stats[link_idx]['combinations_seen']) / \
                      self.exploration_stats[link_idx]['total_combinations']
            exploration_bonus += coverage * self.config.exploration_bonus
        
        reward = sum_rate_value - power_penalty + exploration_bonus
        
        # Info detallada
        info = {
            "sum_rate": sum_rate_value,
            "power_penalty": power_penalty,
            "exploration_bonus": exploration_bonus,
            "power_violations": torch.clamp(power_constr, min=0).sum().item(),
            "mu_k": self.mu_k.clone(),
            "exploration_coverage": [
                len(self.exploration_stats[i]['combinations_seen']) / 
                self.exploration_stats[i]['total_combinations']
                for i in range(self.num_links)
            ],
            "step": self.step_count,
            "data_idx": self.current_data_idx,
            "phi": phi.clone(),
            "rates": rates.clone()
        }
        
        # Criterios de terminación
        terminated = False
        truncated = self.step_count >= 500  # Límite de steps
        
        # Cambiar a siguiente dato ocasionalmente
        if self.step_count > 0 and self.step_count % 100 == 0:
            self.current_data_idx = (self.current_data_idx + 1) % self.dataset_size
        
        self.step_count += 1
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """Mostrar estado actual"""
        if mode == "human":
            print(f"\n=== Step {self.step_count} (Data {self.current_data_idx}) ===")
            
            if hasattr(self, '_last_info'):
                info = self._last_info
                print(f"Sum Rate: {info.get('sum_rate', 0):.3f}")
                print(f"Power Violations: {info.get('power_violations', 0):.3f}")
                print(f"Exploration Bonus: {info.get('exploration_bonus', 0):.3f}")
                
                # Mostrar asignaciones actuales
                if self.current_phi is not None:
                    print("Asignaciones actuales:")
                    phi = self.current_phi[0]  # Remove batch dim
                    for link_idx in range(self.num_links):
                        active_channels = torch.where(phi[link_idx] > 0)[0]
                        if len(active_channels) > 0:
                            channel = active_channels[0].item()
                            power = phi[link_idx, channel].item()
                            print(f"  Link {link_idx}: Canal {channel}, Potencia {power:.2f}")
                        else:
                            print(f"  Link {link_idx}: No transmite")
                
                # Mostrar coverage de exploración
                print("Exploración por link:")
                coverage = info.get('exploration_coverage', [])
                for i, cov in enumerate(coverage):
                    print(f"  Link {i}: {cov:.1%}")
    
    def close(self):
        """Limpiar recursos"""
        pass
    
    def get_exploration_report(self):
        """Obtener reporte detallado de exploración"""
        report = {}
        for link_idx in range(self.num_links):
            stats = self.exploration_stats[link_idx]
            coverage = len(stats['combinations_seen']) / stats['total_combinations']
            report[f'link_{link_idx}'] = {
                'coverage': coverage,
                'combinations_seen': len(stats['combinations_seen']),
                'channels_used': len(stats['channels_used']),
                'powers_used': len(stats['powers_used']),
                'combinations': sorted(list(stats['combinations_seen']))
            }
        return report

# Función helper
def make_wireless_gnn_env(config: WirelessConfig = None):
    """Crear instancia del entorno con tu código integrado"""
    return WirelessGNNEnvironment(config)

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración usando tus parámetros
    config = WirelessConfig(
        building_id=990,
        b5g=False,
        num_links=5,
        num_channels=3,
        num_layers=5,
        K=3,
        p0=4.0,
        synthetic=True
    )
    
    env = make_wireless_gnn_env(config)
    
    print("Entorno creado con tu código integrado!")
    print(f"Dataset size: {env.dataset_size}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Probar episodio
    obs, info = env.reset(seed=42)
    print(f"Observación inicial shape: {obs.shape}")
    
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env._last_info = info  # Para render
        
        print(f"Step {step}: Reward = {reward:.3f}")
        env.render()
        
        if terminated or truncated:
            break
    
    # Mostrar reporte de exploración
    print("\n=== Reporte Final de Exploración ===")
    report = env.get_exploration_report()
    for link, stats in report.items():
        print(f"{link}: {stats['coverage']:.1%} coverage, "
              f"{stats['combinations_seen']}/{stats['combinations'][0] if stats['combinations'] else 0} combinations")
    
    env.close()