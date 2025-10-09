"""
Sistema de Entrenamiento con PPO + GNN Policy
"""

import torch
import numpy as np
import argparse
import pickle
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from network_env import NetworkEnvironment
from gnn_sb3_policy import GNNActorCriticPolicy  # Está en el artifact anterior
from plot_results_torch import plot_results
from utils import load_dataset


class MetricsCallback(BaseCallback):
    """Callback para recolectar métricas y manejar graph_data."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.power_constraints = []
        self.objectives = []
        self.mu_k_history = []
        
        self._current_power_constraints = []
        self._current_objectives = []
        self._current_mu_k = []

    def _on_step(self) -> bool:
        """Llamado después de cada step del entorno."""
        infos = self.locals.get('infos', [])
        
        for info in infos:
            # Recolectar métricas
            if 'power_constraint' in info:
                self._current_power_constraints.append(info['power_constraint'])
            if 'sum_rate' in info:
                self._current_objectives.append(-info['sum_rate'])
            if 'mu_k' in info:
                self._current_mu_k.append(info['mu_k'].copy())
            
            # CRÍTICO: Inyectar graph_data en la política
            if 'graph_data' in info:
                if hasattr(self.model.policy, 'current_graph_data'):
                    self.model.policy.current_graph_data = info['graph_data']
        
        # Detectar fin de episodio
        dones = self.locals.get('dones', [])
        if any(dones):
            if len(self._current_power_constraints) > 0:
                self.power_constraints.append(np.mean(self._current_power_constraints))
                self.objectives.append(np.mean(self._current_objectives))
                self.mu_k_history.append(np.mean(self._current_mu_k, axis=0))
            
            # Reset temporales
            self._current_power_constraints = []
            self._current_objectives = []
            self._current_mu_k = []
        
        return True

    def _on_rollout_end(self) -> None:
        """Logging al final de cada rollout."""
        if self.verbose > 0 and len(self.objectives) > 0:
            print(f"  Episodios completados: {len(self.objectives)}")
            print(f"  Objetivo promedio: {self.objectives[-1]:.4f}")
            print(f"  Restricción promedio: {self.power_constraints[-1]:.4f}")


def train_ppo_gnn(building_id=990, b5g=0, num_links=5, num_channels=3, 
                  num_power_levels=2, num_layers=5, K=3, batch_size=64, 
                  epochs=100, eps=5e-4, mu_lr=1e-4, synthetic=0, 
                  max_antenna_power_dbm=6, sigma=1e-4,
                  # Parámetros PPO
                  total_timesteps=100000, learning_rate=3e-4,
                  n_steps=2048, batch_size_ppo=64, n_epochs_ppo=10,
                  gamma=0.99, gae_lambda=0.95, clip_range=0.2):
    """
    Entrenamiento usando PPO de Stable-Baselines3 con GNN Policy.
    """
    
    # Reproducibilidad
    rn = 267309
    rn1 = 502321
    torch.manual_seed(rn)
    np.random.seed(rn1)
    
    print("=" * 70)
    print("ENTRENAMIENTO PPO + GNN POLICY")
    print("=" * 70)
    
    # ==========================================
    # 1. CARGAR DATASET
    # ==========================================
    dataset = load_dataset(building_id, b5g, num_links, synthetic)
    
    # ==========================================
    # 2. CREAR ENTORNO
    # ==========================================
    print("\nCreando entorno Gymnasium...")
    env = NetworkEnvironment(
        dataset=dataset,
        num_links=num_links,
        num_channels=num_channels,
        num_power_levels=num_power_levels,
        max_steps=len(dataset),
        eps=eps,
        max_antenna_power_dbm=max_antenna_power_dbm,
        sigma=sigma
    )
    
    # Wrapper para monitoreo
    env = Monitor(env)
    
    num_actions = 1 + num_channels * num_power_levels
    
    print("\nConfiguración del entorno:")
    print(f"  Enlaces: {num_links}")
    print(f"  Canales: {num_channels}")
    print(f"  Niveles de potencia: {num_power_levels}")
    print(f"  Acciones por enlace: {num_actions}")
    print(f"  Observación shape: {env.observation_space.shape}")
    print(f"  Acción shape: {env.action_space.shape}")
    
    # ==========================================
    # 3. CONFIGURAR POLÍTICA GNN
    # ==========================================
    print("\nConfigurando GNN Policy...")
    
    policy_kwargs = {
        'num_links': num_links,
        'gnn_hidden_dim': 16,
        'gnn_num_layers': num_layers,
        'gnn_K': K,
        # Configuración de actor-critic (MLP después del GNN)
        'net_arch': [dict(pi=[64, 64], vf=[64, 64])],
        'activation_fn': torch.nn.ReLU
    }
    
    print("  Arquitectura GNN:")
    print(f"    - Hidden dim: {policy_kwargs['gnn_hidden_dim']}")
    print(f"    - Num layers: {policy_kwargs['gnn_num_layers']}")
    print(f"    - K (hops): {policy_kwargs['gnn_K']}")
    print(f"    - Features dim: {num_links * policy_kwargs['gnn_hidden_dim']}")
    print(f"    - Actor/Critic: {policy_kwargs['net_arch']}")
    
    # ==========================================
    # 4. CREAR AGENTE PPO
    # ==========================================
    print("\nCreando agente PPO...")
    
    model = PPO(
        GNNActorCriticPolicy,  # Tu política personalizada
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size_ppo,
        n_epochs=n_epochs_ppo,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/ppo_gnn_{building_id}/"
    )
    
    print("\nConfiguración PPO:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps por rollout: {n_steps}")
    print(f"  Batch size: {batch_size_ppo}")
    print(f"  Epochs por update: {n_epochs_ppo}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE Lambda: {gae_lambda}")
    print(f"  Clip range: {clip_range}")
    
    # ==========================================
    # 5. CALLBACK PARA MÉTRICAS
    # ==========================================
    metrics_callback = MetricsCallback(verbose=1)
    
    # ==========================================
    # 6. ENTRENAMIENTO
    # ==========================================
    print("\n" + "=" * 70)
    print(f"INICIANDO ENTRENAMIENTO - {total_timesteps} timesteps")
    print("=" * 70)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=metrics_callback,
        progress_bar=True
    )
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 70)
    
    # ==========================================
    # 7. GUARDAR RESULTADOS
    # ==========================================
    print("\nGuardando resultados...")
    
    # Crear directorio de resultados
    results_dir = f"./results/ppo_gnn_building_{building_id}_b5g_{b5g}/"
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar modelo de SB3
    model_path = os.path.join(results_dir, "ppo_model.zip")
    model.save(model_path)
    print(f"✓ Modelo PPO guardado en: {model_path}")
    
    # Guardar pesos de la GNN por separado (opcional, útil para análisis)
    gnn_weights_path = os.path.join(results_dir, "gnn_weights.pth")
    torch.save(model.policy.features_extractor.state_dict(), gnn_weights_path)
    print(f"✓ Pesos GNN guardados en: {gnn_weights_path}")
    
    # Guardar métricas
    metrics = {
        'objectives': metrics_callback.objectives,
        'power_constraints': metrics_callback.power_constraints,
        'mu_k_history': metrics_callback.mu_k_history
    }
    
    metrics_path = os.path.join(results_dir, "training_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Métricas guardadas en: {metrics_path}")
    
    # Graficar resultados
    try:
        plot_results(
            building_id=building_id,
            b5g=b5g,
            normalized_psi=None,
            normalized_psi_values=None,
            num_layers=num_layers,
            K=K,
            batch_size=batch_size,
            epochs=len(metrics_callback.objectives),
            rn=rn,
            rn1=rn1,
            eps=eps,
            mu_lr=learning_rate,
            objective_function_values=metrics_callback.objectives,
            power_constraint_values=metrics_callback.power_constraints,
            loss_values=[],
            mu_k_values=metrics_callback.mu_k_history,
            train=True
        )
        print(f"✓ Gráficos guardados en: {results_dir}")
    except Exception as e:
        print(f"⚠ No se pudieron generar gráficos: {e}")
    
    # ==========================================
    # 8. EVALUACIÓN FINAL
    # ==========================================
    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL (100 steps)")
    print("=" * 70)
    
    obs, info = env.reset()
    total_reward = 0
    eval_objectives = []
    eval_power_constraints = []
    
    for step in range(100):
        # Inyectar graph_data antes de predecir
        if 'graph_data' in info:
            model.policy.current_graph_data = info['graph_data']
        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        if 'sum_rate' in info:
            eval_objectives.append(-info['sum_rate'])
        if 'power_constraint' in info:
            eval_power_constraints.append(info['power_constraint'])
        
        if done or truncated:
            obs, info = env.reset()
    
    print(f"\nResultados de evaluación:")
    print(f"  Reward promedio: {total_reward / 100:.4f}")
    print(f"  Objetivo promedio: {np.mean(eval_objectives):.4f}")
    print(f"  Restricción promedio: {np.mean(eval_power_constraints):.4f}")
    
    return results_dir, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenamiento PPO + GNN para redes inalámbricas'
    )
    
    # Parámetros del entorno
    parser.add_argument('--building_id', type=int, default=990)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=5)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_power_levels', type=int, default=2)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--max_antenna_power_dbm', type=int, default=6)
    
    # Parámetros de la GNN
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--eps', type=float, default=5e-4)
    
    # Parámetros de PPO
    parser.add_argument('--total_timesteps', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size_ppo', type=int, default=64)
    parser.add_argument('--n_epochs_ppo', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("CONFIGURACIÓN DE ENTRENAMIENTO")
    print("=" * 70)
    for arg, value in vars(args).items():
        print(f"  {arg:25s}: {value}")
    print("=" * 70 + "\n")
    
    results_dir, model = train_ppo_gnn(
        building_id=args.building_id,
        b5g=args.b5g,
        num_links=args.num_links,
        num_channels=args.num_channels,
        num_power_levels=args.num_power_levels,
        num_layers=args.num_layers,
        K=args.k,
        eps=args.eps,
        synthetic=args.synthetic,
        max_antenna_power_dbm=args.max_antenna_power_dbm,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size_ppo=args.batch_size_ppo,
        n_epochs_ppo=args.n_epochs_ppo,
        gamma=args.gamma
    )
    
    print(f"\n✓ COMPLETADO - Resultados en: {results_dir}")