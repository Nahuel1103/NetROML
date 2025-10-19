import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchrl.data import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# ==== 1) Wrapper del dataset como GymEnv ====
class GraphDatasetEnv(GymEnv):
    def __init__(self, dataset, n_actions):
        # inicializamos con dummy GymEnv, solo necesitamos step/reset
        super().__init__("CartPole-v1")  # no se usa internamente
        self.dataset = dataset
        self.n_actions = n_actions
        self.current_idx = 0
        
        # acción por nodo
        n_nodes = dataset[0].num_nodes
        import gymnasium as gym
        self.action_space = gym.spaces.MultiDiscrete([n_actions]*n_nodes)
        self.observation_space = None  # se maneja en policy

    def reset(self, seed=None, options=None):
        self.current_idx = 0
        return self.dataset[self.current_idx], {}

    def step(self, action):
        reward = torch.rand(1).item()  # ejemplo, reemplazar por función real
        self.current_idx += 1
        done = self.current_idx >= len(self.dataset)
        next_obs = self.dataset[self.current_idx] if not done else None
        return next_obs, reward, done, False, {}

# ==== 2) Política GNN adaptada a TensorDict ====
class GNNActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_actions):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(self, td: TensorDict) -> TensorDict:
        data: Data = td["observation"]      # recibe el grafo
        x, edge_index = data.x, data.edge_index
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        logits = self.head(h)                # [N_nodes, n_actions]

        dist = Categorical(logits=logits)
        action = dist.sample()
        td.set("action", action)
        td.set("log_prob", dist.log_prob(action).sum())  # sum sobre nodos
        return td

# ==== 3) Inicializar entorno y policy ====
dataset = [...]  # tu lista de torch_geometric.data.Data
env_fn = lambda: GraphDatasetEnv(dataset, n_actions=3)
policy = GNNActor(in_dim=dataset[0].num_node_features, hidden_dim=16, n_actions=3)

# ==== 4) Collector ====
collector = SyncDataCollector(
    create_env_fn=env_fn,
    policy=policy,
    frames_per_batch=16,    # pasos por batch
    total_frames=100        # total de pasos
)

# ==== 5) Optimizer ====
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# ==== 6) Loop de entrenamiento (alto nivel) ====
gamma = 0.99
for batch_td in collector:  # cada batch es un TensorDict
    # batch_td["log_prob"] -> log_prob por entorno
    rewards = batch_td["reward"]              # [B]
    
    # calcular retornos descontados
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # REINFORCE loss
    loss = -(batch_td["log_prob"] * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Batch loss: {loss.item():.4f}, reward mean: {rewards.mean():.4f}")
