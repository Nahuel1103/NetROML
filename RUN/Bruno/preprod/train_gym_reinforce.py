import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from envs import APNetworkEnv
from gnn_policy import GNNActorCriticPolicy
from utils import get_gnn_inputs, graphs_to_tensor_synthetic
from realtime_plotter import TrainingVisualizer

def train_reinforce():
    # --- Hyperparameters ---
    n_APs = 5
    num_channels = 3
    n_power_levels = 2
    learning_rate = 1e-4
    gamma = 0.99
    num_episodes = 500  # Adjust as needed
    
    # --- Data Loading ---
    # Using synthetic data for now (same as train_gym.py)
    # Ensure base_path points to where your data is, or remove it to use default
    x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
        num_links=n_APs, num_features=1, b5g=False, building_id=990,
        base_path='/home/bruno/Proyecto/NetROML/RUN/Bruno/preprod/data/' 
    )
    
    dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
    # Batch size 1 because the environment processes one graph at a time
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    def dataloader_generator(loader):
        while True:
            for data in loader:
                # data.matrix is [1, N, N] because batch_size=1
                # Convert to numpy [N, N]
                yield data.matrix.squeeze(0).numpy()

    H_iterator = dataloader_generator(dataloader)

    # --- Environment ---
    env = APNetworkEnv(
        n_APs=n_APs,
        num_channels=num_channels,
        n_power_levels=n_power_levels,
        P0=4,
        Pmax=0.7,
        max_steps=50,
        H_iterator=H_iterator
    )

    # --- Policy ---
    # We need a dummy lr_schedule because SB3 Policy expects it
    def lr_schedule(progress_remaining):
        return learning_rate

    policy = GNNActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule,
        gnn_hidden_dim=32,
        gnn_num_layers=3,
        K=3
    )

    # We only optimize the policy parameters
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # --- Visualizer ---
    visualizer = TrainingVisualizer(n_APs, num_channels)

    # --- Training Loop ---
    print("Starting REINFORCE training...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        terminated = False
        truncated = False
        
        episode_probs_accum = []
        episode_off_probs_accum = []

        # Collect Episode
        while not (terminated or truncated):
            # Manual conversion to tensor for the policy input
            obs_tensor = {
                key: torch.tensor(val).unsqueeze(0) if isinstance(val, np.ndarray) else val 
                for key, val in obs.items()
            }
            
            # Get action and log prob
            actions, _, log_prob = policy.forward(obs_tensor)
            
            # --- Probability Extraction for Visualization ---
            # We need to get the distribution to extract probabilities
            # Forward calls _get_action_dist_from_latent internally but doesn't return it directly in a way we can easily access probs without re-running parts or modifying forward.
            # However, we can re-use the logic or just call a helper if we had one.
            # Let's peek at how `forward` does it: it gets logits.
            # We can use `policy.get_distribution(obs_tensor)` if implemented, or manually:
            # The policy `forward` returns actions, value, log_probs.
            # To get raw probs, we might need to access the distribution.
            # Let's use `policy.evaluate_actions` which returns entropy/log_prob but not raw probs directly?
            # Actually, `policy.get_distribution(obs_tensor)` is a standard SB3 method.
            
            # But `get_distribution` expects `obs` as tensor? Yes.
            # And it returns a Distribution object.
            dist = policy.get_distribution(obs_tensor)
            # dist is MultiCategorical.
            # We want the probabilities.
            # SB3 MultiCategorical distribution wraps PyTorch Categorical or similar.
            # It has `distribution.distribution` which is a list of Categoricals (if MultiDiscrete) or a single Categorical?
            # Wait, SB3 MultiCategorical uses a single Categorical with flattened actions usually?
            # Let's check `gnn_policy.py`. It uses `_get_action_dist_from_latent`.
            # And `logits` are [batch, n_APs * n_actions].
            # So `dist` corresponds to these logits.
            # We can get probs from `dist.distribution.probs` if it's a Categorical.
            
            # Accessing internal distribution
            # The SB3 MultiCategorical distribution stores `self.distribution` which is a `torch.distributions.Categorical`.
            # Its probs will be [batch, total_actions_flattened] or similar.
            
            # Let's try to get logits/probs directly.
            # Since we are in the loop, we can just do:
            # features = policy.features_extractor(obs_tensor)
            # ... (replicate forward logic to get logits) ...
            # OR, simpler:
            # The `dist` object has `log_prob(actions)`.
            # It might not expose `probs` directly if it's a wrapper.
            # But `dist.distribution` should be the PyTorch distribution.
            
            # dist.distribution is a list of Categoricals for MultiDiscrete
            probs_list = [d.probs for d in dist.distribution] # List of [batch, 7]
            all_probs = torch.stack(probs_list, dim=1) # [batch, n_APs, 7]
            
            # Reshape to [n_APs, n_actions] (batch=1)
            all_probs = all_probs.view(n_APs, -1)
            
            # Extract channel probs and off probs
            # Action 0 is OFF
            off_probs = all_probs[:, 0] # [n_APs]
            
            # Active actions: 1..end
            active_probs = all_probs[:, 1:] # [n_APs, num_channels * num_power_levels]
            active_probs = active_probs.view(n_APs, num_channels, n_power_levels)
            channel_probs = active_probs.sum(dim=2) # [n_APs, num_channels]
            
            episode_probs_accum.append(channel_probs.detach().cpu().numpy())
            episode_off_probs_accum.append(off_probs.detach().cpu().numpy())
            # ------------------------------------------------
            
            # Action to numpy for env
            action_numpy = actions.cpu().numpy()[0] # [0] because batch size 1
            
            obs, reward, terminated, truncated, info = env.step(action_numpy)
            
            log_probs.append(log_prob)
            rewards.append(reward)

        # --- Calculate Returns (Monte Carlo) ---
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # --- Update Policy ---
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Update Visualizer ---
        avg_reward = np.mean(rewards)
        avg_loss = loss.item() / len(rewards) # Average loss per step roughly
        
        avg_probs_episode = np.mean(np.array(episode_probs_accum), axis=0)
        avg_off_probs_episode = np.mean(np.array(episode_off_probs_accum), axis=0)
        
        visualizer.update(avg_reward, avg_loss, avg_probs_episode, avg_off_probs_episode)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {sum(rewards):.2f} | Loss: {loss.item():.2f}")

    print("Training finished.")
    visualizer.close()
    
    # Save model
    torch.save(policy.state_dict(), "reinforce_gnn_policy.pth")
    print("Model saved to reinforce_gnn_policy.pth")

if __name__ == "__main__":
    train_reinforce()
