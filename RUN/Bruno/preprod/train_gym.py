import numpy as np
import torch
from stable_baselines3 import PPO
from gnn_policy import GNNActorCriticPolicy
from envs import APNetworkEnv
from utils import load_channel_matrix


# def channel_iterator(num_graphs=1000, n_APs=5):
#     for _ in range(num_graphs):
#         H = np.random.rand(n_APs, n_APs).astype(np.float32)
#         np.fill_diagonal(H, H.diagonal() * 2)
#         yield H

def train_gnn_policy():
    n_APs = 5
    num_channels = 3
    n_power_levels = 2

    H_iterator = load_channel_matrix(
            building_id=990,
            b5g=False,  # 2.4 GHz
            num_links=n_APs,
            synthetic=False,  # Cambiar a True para usar datos sintéticos
            shuffle=True,
            repeat=True  # Reinicia automáticamente el iterador
        )

    env = APNetworkEnv(
        n_APs=n_APs,
        num_channels=num_channels,
        n_power_levels=n_power_levels,
        P0=4,
        Pmax=0.7,
        max_steps=50,
        H_iterator=H_iterator
    )

    policy_kwargs = dict(gnn_hidden_dim=32,
                         gnn_num_layers=3,
                         K=3)

    model = PPO(GNNActorCriticPolicy,
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                verbose=1)

    model.learn(total_timesteps=50000)
    model.save("ppo_gnn_policy")


if __name__ == "__main__":
    train_gnn_policy()
