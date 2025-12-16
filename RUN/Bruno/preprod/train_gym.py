import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gnn_policy import GNNActorCriticPolicy
from envs import APNetworkEnv
from utils import load_channel_matrix, get_gnn_inputs, graphs_to_tensor, graphs_to_tensor_synthetic
from torch_geometric.loader import DataLoader
from realtime_plotter import TrainingVisualizer


class VisualizerCallback(BaseCallback):
    def __init__(self, num_links, num_channels, verbose=0):
        super(VisualizerCallback, self).__init__(verbose)
        self.num_links = num_links
        self.num_channels = num_channels
        self.visualizer = None
        
        # Accumulators for the current episode/epoch
        self.episode_probs_accum = []
        self.episode_off_probs_accum = []
        self.episode_rewards = []
        
    def _on_training_start(self) -> None:
        self.visualizer = TrainingVisualizer(self.num_links, self.num_channels)
        
    def _on_step(self) -> bool:
        # Access the current observation
        # SB3 stores current obs in self.locals['new_obs'] or self.locals['obs_tensor']
        # Let's use the policy to get the distribution
        
        # self.locals['obs_tensor'] is usually available
        obs_tensor = self.locals.get('obs_tensor')
        if obs_tensor is None:
            return True
            
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            
            # dist.distribution is a list of Categoricals for MultiDiscrete
            probs_list = [d.probs for d in dist.distribution] # List of [batch, 7]
            all_probs = torch.stack(probs_list, dim=1) # [batch, n_APs, 7]
            
            # Assuming batch size 1 for simplicity in visualization logic, or we average over batch
            # If batch > 1, we take mean
            
            # Reshape to [batch, n_APs, n_actions]
            batch_size = all_probs.shape[0]
            # all_probs is already [batch, n_APs, 7]
            
            # Off probs: index 0
            off_probs = all_probs[:, :, 0] # [batch, n_APs]
            
            # Active probs
            active_probs = all_probs[:, :, 1:] # [batch, n_APs, num_channels * num_power_levels]
            # Reshape to separate channels and power levels
            # We need to know num_power_levels. 
            # num_actions = 1 + num_channels * num_power_levels
            # active_actions = num_channels * num_power_levels
            active_actions_count = active_probs.shape[-1]
            num_power_levels = active_actions_count // self.num_channels
            
            active_probs = active_probs.view(batch_size, self.num_links, self.num_channels, num_power_levels)
            channel_probs = active_probs.sum(dim=3) # [batch, n_APs, num_channels]
            
            # Average over batch
            avg_off = off_probs.mean(dim=0).cpu().numpy()
            avg_channel = channel_probs.mean(dim=0).cpu().numpy()
            
            self.episode_probs_accum.append(avg_channel)
            self.episode_off_probs_accum.append(avg_off)
            
            # Track rewards
            # self.locals['rewards'] contains rewards for the current step
            rewards = self.locals.get('rewards')
            if rewards is not None:
                self.episode_rewards.append(np.mean(rewards))
                
        return True

    def _on_rollout_end(self) -> None:
        # Called before updating the policy
        # Update visualizer here
        
        if not self.episode_probs_accum:
            return

        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        # Try to get loss. It's not easily available in _on_rollout_end because update hasn't happened yet.
        # But we can get the loss from the PREVIOUS update if logged.
        # Or we can just plot 0 for now, or use the logger.
        # self.logger.name_to_value['train/loss'] might exist
        loss = 0
        if hasattr(self.logger, 'name_to_value') and 'train/loss' in self.logger.name_to_value:
            loss = self.logger.name_to_value['train/loss']
            
        avg_probs = np.mean(np.array(self.episode_probs_accum), axis=0)
        avg_off = np.mean(np.array(self.episode_off_probs_accum), axis=0)
        
        self.visualizer.update(avg_reward, loss, avg_probs, avg_off)
        
        # Reset accumulators
        self.episode_probs_accum = []
        self.episode_off_probs_accum = []
        self.episode_rewards = []

    def _on_training_end(self) -> None:
        if self.visualizer:
            self.visualizer.close()


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

    # Create callback
    vis_callback = VisualizerCallback(n_APs, num_channels)

    model.learn(total_timesteps=50000, callback=vis_callback)
    model.save("ppo_gnn_policy")


if __name__ == "__main__":
    train_gnn_policy()
