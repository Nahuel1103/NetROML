import numpy as np
import torch
from stable_baselines3 import PPO
from gnn_policy import GNNActorCriticPolicy
from envs import APNetworkEnv
from utils import load_channel_matrix, get_gnn_inputs, graphs_to_tensor, graphs_to_tensor_synthetic
from torch_geometric.loader import DataLoader


# def channel_iterator(num_graphs=1000, n_APs=5):
#     for _ in range(num_graphs):
#         H = np.random.rand(n_APs, n_APs).astype(np.float32)
#         np.fill_diagonal(H, H.diagonal() * 2)
#         yield H

def train_gnn_policy():
    n_APs = 5
    num_channels = 3
    n_power_levels = 2

    # Cargar datos y crear DataLoader
    # x_tensor, channel_matrix_tensor = graphs_to_tensor(
    #     train=True, num_links=n_APs, num_features=1, b5g=False, building_id=990
    # )
    # Para usar datos sintéticos (comentar lo de arriba y descomentar esto si se desea)
    # x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
    #      num_links=n_APs, num_features=1, b5g=False, building_id=990,
    #      base_path='/home/bruno/Proyecto/NetROML/RUN/Bruno/preprod/data/'
    # )
    x_tensor, channel_matrix_tensor = graphs_to_tensor(
         num_links=n_APs, num_features=1, b5g=False, building_id=990,
        )
    
    # Crear DataLoader
    dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
    # Batch size 1 porque el entorno procesa un grafo a la vez por ahora
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    def dataloader_generator(loader):
        while True:
            for data in loader:
                # data.matrix es [1, N, N] porque batch_size=1
                # Lo convertimos a numpy [N, N]
                yield data.matrix.squeeze(0).numpy()

    H_iterator = dataloader_generator(dataloader)

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
