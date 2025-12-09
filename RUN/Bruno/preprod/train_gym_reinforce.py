import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from envs import APNetworkEnv
from gnn_policy import GNNActorCriticPolicy
from utils import get_gnn_inputs, graphs_to_tensor_synthetic

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

    # We only optimize the policy parameters (Actor and Critic share weights in this implementation? 
    # Actually GNNActorCriticPolicy has separate heads but shared GNN extractor.
    # REINFORCE typically only updates the Actor, but if we update the whole network 
    # based on policy loss, it should be fine. The Value head will just drift or be ignored.)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # --- Training Loop ---
    print("Starting REINFORCE training...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        terminated = False
        truncated = False
        
        # Collect Episode
        while not (terminated or truncated):
            # Get action from policy
            # forward returns: actions, value, log_probs
            # We need to convert obs to tensor dict if not already handled by policy?
            # SB3 policy expects dict of tensors or numpy arrays. 
            # Let's check GNNFeaturesExtractor.forward: it handles "H" and "mu".
            # It expects tensors. But SB3 usually handles numpy->tensor conversion in `predict`.
            # Here we are calling `forward` directly. We might need to convert obs.
            
            # Manual conversion to tensor for the policy input
            obs_tensor = {
                key: torch.tensor(val).unsqueeze(0) if isinstance(val, np.ndarray) else val 
                for key, val in obs.items()
            }
            # Handle dimensions: H needs to be [batch, N, N], mu [batch, N]
            # If env returns H as [N, N], unsqueeze adds batch dim -> [1, N, N]
            
            actions, _, log_prob = policy.forward(obs_tensor)
            
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
        # Optional: Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # --- Update Policy ---
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            # REINFORCE objective: maximize log_prob * G
            # Loss: minimize -(log_prob * G)
            loss += -log_prob * G
        
        # Average loss over the episode (optional, but good for magnitude)
        # loss = loss / len(rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {sum(rewards):.2f} | Loss: {loss.item():.2f}")

    print("Training finished.")
    
    # Save model
    torch.save(policy.state_dict(), "reinforce_gnn_policy.pth")
    print("Model saved to reinforce_gnn_policy.pth")

if __name__ == "__main__":
    train_reinforce()
