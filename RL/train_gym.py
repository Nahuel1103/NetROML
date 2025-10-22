import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import argparse
import pickle
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from RUN.utils import load_dataset
from network_env import NetworkEnvironment
from gnn import GNN 
from utils import load_dataset
from plot_results_torch import plot_results

# class MetricsCallback(BaseCallback):
#     """Callback para recolectar métricas y manejar graph_data."""
    
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.episode_rewards = []
#         self.power_constraints = []
#         self.objectives = []
#         self.mu_k_history = []
        
#         self._current_power_constraints = []
#         self._current_objectives = []
#         self._current_mu_k = []

#     def _on_step(self) -> bool:
#         """Llamado después de cada step del entorno."""
#         infos = self.locals.get('infos', [])
        
#         for info in infos:
#             # Recolectar métricas
#             if 'power_constraint' in info:
#                 self._current_power_constraints.append(info['power_constraint'])
#             if 'sum_rate' in info:
#                 self._current_objectives.append(-info['sum_rate'])
#             if 'mu_k' in info:
#                 self._current_mu_k.append(info['mu_k'].copy())
            
#             # CRÍTICO: Inyectar graph_data en la política
#             if 'graph_data' in info:
#                 if hasattr(self.model.policy, 'current_graph_data'):
#                     self.model.policy.current_graph_data = info['graph_data']
        
#         # Detectar fin de episodio
#         dones = self.locals.get('dones', [])
#         if any(dones):
#             if len(self._current_power_constraints) > 0:
#                 self.power_constraints.append(np.mean(self._current_power_constraints))
#                 self.objectives.append(np.mean(self._current_objectives))
#                 self.mu_k_history.append(np.mean(self._current_mu_k, axis=0))
            
#             # Reset temporales
#             self._current_power_constraints = []
#             self._current_objectives = []
#             self._current_mu_k = []
        
#         return True

#     def _on_rollout_end(self) -> None:
#         """Logging al final de cada rollout."""
#         if self.verbose > 0 and len(self.objectives) > 0:
#             print(f"  Episodios completados: {len(self.objectives)}")
#             print(f"  Objetivo promedio: {self.objectives[-1]:.4f}")
#             print(f"  Restricción promedio: {self.power_constraints[-1]:.4f}")


def train(building_id=990, b5g=0, num_layers=5, num_links=5, num_channels=3, 
                  num_power_levels=2, K=3, batch_size=64, 
                  epochs=100, eps=5e-4, mu_lr=1e-4, synthetic=0, 
                  max_antenna_power_dbm=6, sigma=1e-4):

    input_dim = 1
    hidden_dim = 1
    output_dim = 1 + num_channels * num_power_levels

    # ---------- 1) Cargar datos ----------

    data = load_dataset(building_id, b5g, num_links, batch_size, synthetic)

    channel_matrix = data.matrix
    channel_matrix = channel_matrix.view(batch_size, num_links, num_links)

    # ---------- 2) Crear Environment ----------

    env = NetworkEnvironment(channel_matrix=channel_matrix)

    # Llamar reset() para crear el iterador interno
    obs, info = env.reset()

    gnn_model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                    num_layers=num_layers, dropout=False, K=K).to(device)
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)

 



    for name, param in gnn_model.named_parameters():
        param.requires_grad = True
        if param.requires_grad:
            print(name, param.data)

    

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            break

            epochs = epochs
    for ep in range(start_epoch, epochs):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        for step_idx in range(env.max_steps):
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            psi = psi.view(batch_size, num_links, output_dim)
            # Adaptá la forma si tu GNN devuelve [L, output_dim] o [1, L, output_dim]
            # Ejemplo: tomar softmax por nodo / enlace
            logits = psi.view(num_links, -1)  # [L, output_dim]
            probs = torch.softmax(logits, dim=-1)
            # elegir acción por enlace (greedy o sample)
            actions_per_link = torch.argmax(probs, dim=-1).cpu().numpy()  # array length L

            # mapear acciones_per_link a formato que espera env (lista ints)
            action = [int(a) for a in actions_per_link]  # asegúrate que el mapeo coincide con env (0=no_tx, 1..C*P)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward