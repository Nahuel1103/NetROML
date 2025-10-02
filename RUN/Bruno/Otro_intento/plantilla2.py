# Environment wrapper

import gymnasium as gym
from torchrl.envs import GymEnv

class GraphDatasetEnv(GymEnv):
    def __init__(self, dataset, n_actions):
        super().__init__("CartPole-v1")  # GymEnv dummy
        self.dataset = dataset
        self.n_actions = n_actions
        self.current_idx = 0
        
        import gymnasium as gym
        self.action_space = gym.spaces.MultiDiscrete([n_actions]*dataset[0].num_nodes)
        self.observation_space = None

    def reset(self, seed=None, options=None):
        self.current_idx = 0
        return self.dataset[self.current_idx], {}

    def step(self, action):
        reward = torch.rand(1).item()  # reemplazar por tu función real
        self.current_idx += 1
        done = self.current_idx >= len(self.dataset)
        next_obs = self.dataset[self.current_idx] if not done else None
        return next_obs, reward, done, False, {}



# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Política GNN (adaptada a TensorDict)


import torch.nn as nn
from torch.distributions import Categorical
from torchrl.data import TensorDict
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_actions):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(self, td: TensorDict) -> TensorDict:
        data: Data = td["observation"]
        x, edge_index = data.x, data.edge_index
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        logits = self.head(h)  # [N_nodes, n_actions]

        dist = Categorical(logits=logits)
        action = dist.sample()
        td.set("action", action)
        td.set("log_prob", dist.log_prob(action).sum())  # sumar sobre nodos
        return td






# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------




# Entrenamiento con TorchRL + REINFORCE

from torchrl.collectors import SyncDataCollector
import torch

# Inicializamos entorno y policy
env_fn = lambda: GraphDatasetEnv(dataset, n_actions=2)
policy = GNNActor(in_dim=3, hidden_dim=16, n_actions=2)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# Collector
collector = SyncDataCollector(
    create_env_fn=env_fn,
    policy=policy,
    frames_per_batch=5,   # pasos por batch
    total_frames=20       # total de pasos
)

gamma = 0.99
for batch_td in collector:
    rewards = batch_td["reward"]  # [B] vector de recompensas

    # calcular retornos descontados
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # REINFORCE loss
    loss = -(batch_td["log_prob"] * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch loss: {loss.item():.4f}, reward mean: {rewards.mean():.4f}")



