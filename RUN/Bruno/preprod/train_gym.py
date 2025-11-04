import numpy as np
import torch
from stable_baselines3 import PPO
from gnn_policy_No_Batches import GNNActorCriticPolicy
from envs import APNetworkEnv


def channel_iterator(num_graphs=1000, n_APs=5):
    for _ in range(num_graphs):
        H = np.random.rand(n_APs, n_APs).astype(np.float32)
        np.fill_diagonal(H, H.diagonal() * 2)
        yield H


def train_gnn_policy():
    n_APs = 5
    num_channels = 3
    n_power_levels = 2

    env = APNetworkEnv(
        n_APs=n_APs,
        num_channels=num_channels,
        n_power_levels=n_power_levels,
        P0=4,
        Pmax=0.7,
        max_steps=50,
        H_iterator=channel_iterator(10000, n_APs)
    )

    policy_kwargs = dict(n_APs=n_APs,
                         gnn_hidden_dim=32,
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
