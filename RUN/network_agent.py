"""
Versión simplificada del agente RL que funciona sin torch-geometric.
Esta versión usa MLP standard para comenzar, luego puedes evolucionarla a GNN.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


# ====================
# 1. ENTORNO PERSONALIZADO
# ====================

class NetworkResourceEnv(gym.Env):
    """
    Entorno de optimización de recursos de red - versión simplificada.
    """
    
    def __init__(self, 
                 num_links: int = 5,
                 num_channels: int = 3,
                 max_antenna_power_dbm: float = 6.0,
                 sigma: float = 1e-4,
                 episode_length: int = 1000,
                 scenario_change_freq: int = 100,
                 channel_matrices: Optional[List[np.ndarray]] = None):
        
        super().__init__()
        
        # Configuración de red
        self.num_links = num_links
        self.num_channels = num_channels
        self.max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
        self.sigma = sigma
        
        # Configuración de episodios
        self.max_episode_steps = episode_length
        self.scenario_change_freq = scenario_change_freq
        self.current_step = 0
        
        # Espacios de acción y observación
        # Acciones: [no_tx, ch0_half, ch1_half, ch2_half, ch0_full, ch1_full, ch2_full] 
        self.action_space = gym.spaces.MultiDiscrete([7] * num_links)
        
        # Observación: matriz H flattened + info adicional
        obs_dim = num_links * num_links + num_links * 2  # H + dual vars + prev actions
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Restricciones de potencia
        self.pmax_per_ap = self.max_antenna_power_mw * 0.8 * np.ones(num_links)
        
        # Variables duales (tu enfoque Lagrangiano)
        self.mu_k = np.ones(num_links)
        self.mu_lr = 5e-4
        
        # Datos de canal
        self.channel_matrices = channel_matrices or self._generate_synthetic_matrices()
        self.current_channel_matrix = None
        self.current_matrix_idx = 0
        
        # Estado del entorno
        self.previous_actions = np.zeros(num_links, dtype=int)
        self.episode_rewards = []
        
    def _generate_synthetic_matrices(self, num_matrices: int = 200) -> List[np.ndarray]:
        """Generar matrices de canal sintéticas para entrenamiento"""
        matrices = []
        np.random.seed(42)
        
        for _ in range(num_matrices):
            # Generar matriz con propiedades realistas
            H = np.random.exponential(scale=0.5, size=(self.num_links, self.num_links))
            
            # Enlaces directos más fuertes
            np.fill_diagonal(H, H.diagonal() * 3)
            
            # Añadir correlación espacial
            for i in range(self.num_links):
                for j in range(self.num_links):
                    if i != j:
                        distance_factor = 1.0 / (1.0 + 0.5 * abs(i - j))
                        H[i, j] *= distance_factor
            
            matrices.append(H)
        
        return matrices
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset de episodio
        self.current_step = 0
        self.current_matrix_idx = np.random.randint(len(self.channel_matrices))
        self.current_channel_matrix = self.channel_matrices[self.current_matrix_idx].copy()
        
        # Reset variables de estado
        self.mu_k = np.ones(self.num_links)
        self.previous_actions = np.zeros(self.num_links, dtype=int)
        self.episode_rewards = []
        
        observation = self._get_observation()
        info = {'matrix_idx': self.current_matrix_idx}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1
        
        # Cambiar escenario periódicamente
        if self.current_step % self.scenario_change_freq == 0:
            self.current_matrix_idx = np.random.randint(len(self.channel_matrices))
            self.current_channel_matrix = self.channel_matrices[self.current_matrix_idx].copy()
        
        # Procesar acciones usando tu lógica original
        phi = self._actions_to_power_allocation(action)
        
        # Calcular tasas usando tu función nuevo_get_rates adaptada
        rates = self._calculate_rates(phi)
        
        # Verificar restricciones de potencia
        power_constraints = self._calculate_power_constraints(phi)
        
        # Actualizar variables duales (tu enfoque Lagrangiano)
        self.mu_k += self.mu_lr * power_constraints
        self.mu_k = np.maximum(self.mu_k, 0.0)
        
        # Calcular recompensa
        reward = self._calculate_reward(rates, power_constraints, action)
        
        # Actualizar estado
        self.previous_actions = action.copy()
        self.episode_rewards.append(reward)
        
        # Terminación
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        
        # Información para análisis
        info = {
            'sum_rate': np.sum(rates),
            'individual_rates': rates,
            'power_constraints': power_constraints,
            'constraint_violations': np.sum(power_constraints > 0),
            'mu_k': self.mu_k.copy(),
            'actions': action.copy()
        }
        
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def _actions_to_power_allocation(self, actions: np.ndarray) -> np.ndarray:
        """
        Convertir acciones discretas a asignación de potencia.
        Usa exactamente tu esquema de 7 acciones.
        """
        phi = np.zeros((self.num_links, self.num_channels))
        
        for link_idx, action in enumerate(actions):
            if action == 0:
                # No transmitir
                continue
            elif 1 <= action <= self.num_channels:
                # Media potencia: acciones 1,2,3 -> canales 0,1,2
                channel_idx = action - 1
                power_level = self.max_antenna_power_mw / 2.0
                phi[link_idx, channel_idx] = power_level
            elif (self.num_channels + 1) <= action <= (2 * self.num_channels):
                # Potencia completa: acciones 4,5,6 -> canales 0,1,2
                channel_idx = action - (self.num_channels + 1)
                power_level = self.max_antenna_power_mw
                phi[link_idx, channel_idx] = power_level
        
        return phi
    
    def _calculate_rates(self, phi: np.ndarray) -> np.ndarray:
        """
        Calcular tasas de transmisión usando tu modelo de interferencias.
        Adaptación de tu función nuevo_get_rates.
        """
        H = self.current_channel_matrix
        rates = np.zeros(self.num_links)
        
        for link_idx in range(self.num_links):
            link_rate = 0.0
            
            for channel_idx in range(self.num_channels):
                if phi[link_idx, channel_idx] > 0:
                    # Señal útil
                    signal_power = H[link_idx, link_idx] * phi[link_idx, channel_idx]
                    
                    # Interferencia intra-canal de otros enlaces
                    interference = 0.0
                    for other_link in range(self.num_links):
                        if other_link != link_idx:
                            interference += H[link_idx, other_link] * phi[other_link, channel_idx]
                    
                    # SINR y tasa
                    denominator = interference + self.sigma
                    if denominator > 0:
                        sinr = signal_power / denominator
                        link_rate += np.log(1 + sinr)
            
            rates[link_idx] = link_rate
        
        return rates
    
    def _calculate_power_constraints(self, phi: np.ndarray) -> np.ndarray:
        """Calcular violaciones de restricciones de potencia por AP"""
        power_per_ap = np.sum(phi, axis=1)  # Suma por canales
        return power_per_ap - self.pmax_per_ap
    
    def _calculate_reward(self, rates: np.ndarray, power_constraints: np.ndarray, 
                         actions: np.ndarray) -> float:
        """
        Función de recompensa multi-objetivo inspirada en tu enfoque Lagrangiano.
        """
        # Objetivo principal: maximizar suma de tasas
        sum_rate = np.sum(rates)
        
        # Penalización por restricciones (tu enfoque dual)
        power_violations = np.maximum(power_constraints, 0)
        lagrangian_penalty = np.sum(self.mu_k * power_violations)
        
        # Bonificación por eficiencia energética
        total_power = np.sum([self._get_power_from_action(a) for a in actions])
        efficiency_bonus = 0.1 * (sum_rate / (total_power + 1e-6) if total_power > 0 else 0)
        
        # Penalización leve por cambios bruscos (estabilidad)
        action_changes = np.sum(actions != self.previous_actions)
        stability_penalty = 0.01 * action_changes
        
        # Recompensa final
        reward = sum_rate - lagrangian_penalty + efficiency_bonus - stability_penalty
        
        return reward
    
    def _get_power_from_action(self, action: int) -> float:
        """Obtener potencia usada por una acción específica"""
        if action == 0:
            return 0.0
        elif 1 <= action <= self.num_channels:
            return self.max_antenna_power_mw / 2.0
        else:
            return self.max_antenna_power_mw
    
    def _get_observation(self) -> np.ndarray:
        """
        Construir observación del estado actual.
        Incluye matriz H, variables duales, y acciones previas.
        """
        # Normalizar matriz de canal
        h_flat = self.current_channel_matrix.flatten()
        h_norm = (h_flat - np.mean(h_flat)) / (np.std(h_flat) + 1e-8)
        
        # Variables duales normalizadas
        mu_norm = self.mu_k / (np.max(self.mu_k) + 1e-8)
        
        # Acciones previas normalizadas
        prev_actions_norm = self.previous_actions / 6.0  # Max action value is 6
        
        # Concatenar observación
        obs = np.concatenate([h_norm, mu_norm, prev_actions_norm])
        
        return obs.astype(np.float32)


# ====================
# 2. ARQUITECTURA DE RED NEURONAL SIMPLIFICADA
# ====================

class NetworkFeatureExtractor(BaseFeaturesExtractor):
    """
    Extractor de características MLP simple para comenzar.
    Una vez que esto funcione, puedes evolucionar a GNN.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # Red neuronal feed-forward
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        # Inicialización
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class NetworkPolicy(ActorCriticPolicy):
    """
    Política personalizada que usa nuestro extractor de características.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, 
            features_extractor_class=NetworkFeatureExtractor,
            **kwargs
        )


# ====================
# 3. CALLBACK PARA MONITOREO SIMPLIFICADO
# ====================

class SimpleTrainingCallback(BaseCallback):
    """
    Callback simplificado para monitorear el entrenamiento.
    """
    
    def __init__(self, eval_env, eval_freq: int = 5000, save_path: str = "./results/"):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        
        # Métricas básicas
        self.training_rewards = []
        self.sum_rates = []
        self.constraint_violations = []
        self.eval_results = []
        
    def _on_step(self) -> bool:
        # Recopilar información básica
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            if 'sum_rate' in info:
                self.sum_rates.append(info['sum_rate'])
            if 'constraint_violations' in info:
                self.constraint_violations.append(info['constraint_violations'])
        
        # Evaluación periódica
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=10
            )
            
            self.eval_results.append({
                'step': self.n_calls,
                'mean_reward': mean_reward,
                'std_reward': std_reward
            })
            
            print(f"Paso {self.n_calls}: Reward = {mean_reward:.3f} ± {std_reward:.3f}")
        
        return True
    
    def save_results(self):
        """Guardar resultados básicos"""
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        results = {
            'sum_rates': self.sum_rates,
            'constraint_violations': self.constraint_violations,
            'eval_results': self.eval_results
        }
        
        with open(f"{self.save_path}/training_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Plot simple
        if len(self.sum_rates) > 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            # Moving average para suavizar
            window = min(100, len(self.sum_rates) // 10)
            if window > 1:
                smoothed = np.convolve(self.sum_rates, np.ones(window)/window, mode='valid')
                plt.plot(smoothed)
            else:
                plt.plot(self.sum_rates)
            plt.title('Sum Rate Durante Entrenamiento')
            plt.xlabel('Pasos')
            plt.ylabel('Sum Rate')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            if len(self.constraint_violations) > 0:
                window = min(100, len(self.constraint_violations) // 10)
                if window > 1:
                    smoothed = np.convolve(self.constraint_violations, np.ones(window)/window, mode='valid')
                    plt.plot(smoothed)
                else:
                    plt.plot(self.constraint_violations)
            plt.title('Violaciones de Restricciones')
            plt.xlabel('Pasos')
            plt.ylabel('Número de Violaciones')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{self.save_path}/training_progress.png")
            plt.close()


# ====================
# 4. FUNCIÓN PRINCIPAL SIMPLIFICADA
# ====================

def create_and_train_agent(
    num_links: int = 5,
    num_channels: int = 3,
    total_timesteps: int = 50000,
    save_path: str = "./results/simple_network_agent/"
):
    """
    Función principal para crear y entrenar el agente RL simplificado.
    """
    
    print(f"Creando agente RL para red de {num_links} enlaces y {num_channels} canales")
    
    # Crear entornos
    def make_env():
        return NetworkResourceEnv(
            num_links=num_links,
            num_channels=num_channels,
            max_antenna_power_dbm=6.0,
            sigma=1e-4,
            episode_length=500,  # Episodios más cortos para empezar
            scenario_change_freq=50  # Cambios más frecuentes
        )
    
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    # Modelo PPO simplificado
    model = PPO(
        NetworkPolicy,
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard/"
    )
    
    # Callback simplificado
    callback = SimpleTrainingCallback(
        eval_env=eval_env,
        eval_freq=2000,
        save_path=save_path
    )
    
    # Entrenar
    print("Iniciando entrenamiento...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name="simple_network_optimization"
    )
    
    # Guardar modelo y resultados
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save(f"{save_path}/final_model")
    callback.save_results()
    
    print(f"Entrenamiento completado. Modelo guardado en: {save_path}")
    
    # Evaluación final
    final_reward, final_std = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Evaluación final: {final_reward:.3f} ± {final_std:.3f}")
    
    return model, callback


# ====================
# 5. VALIDACIÓN RÁPIDA
# ====================

def quick_test():
    """Función para validar rápidamente que todo funciona"""
    print("Ejecutando test rápido...")
    
    env = NetworkResourceEnv(num_links=3, num_channels=2)
    
    # Test básico
    obs, info = env.reset()
    print(f"✓ Reset OK - Observación shape: {obs.shape}")
    
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if i == 0:
            print(f"✓ Step OK - Reward: {reward:.3f}, Sum rate: {info['sum_rate']:.3f}")
    
    print(f"✓ Test completado - Reward promedio: {total_reward/10:.3f}")
    print("El entorno funciona correctamente!")


# ====================
# 6. SCRIPT PRINCIPAL
# ====================

if __name__ == "__main__":
    # Primero ejecutar test rápido
    quick_test()
    
    print("\n" + "="*50)
    
    # Configuración del experimento
    config = {
        'num_links': 5,
        'num_channels': 3,
        'total_timesteps': 50000,
        'save_path': "./results/simple_network_agent_v1/"
    }
    
    # Crear y entrenar agente
    model, callback = create_and_train_agent(**config)
    
    print("¡Agente entrenado exitosamente!")
    print(f"Revisa los resultados en: {config['save_path']}")