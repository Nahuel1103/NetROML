import torch
from torch_geometric.nn import GCNConv

class GNNPolicy(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_features)

    def forward(self, x, edge_index):
        # x: features de nodos (por ejemplo, potencias, canales, etc.)
        # edge_index: conectividad del grafo
        h = torch.relu(self.conv1(x, edge_index))
        logits = self.conv2(h, edge_index)
        return logits  # luego se usa para decidir acciones

    def get_action(self, x, edge_index):
        logits = self.forward(x, edge_index)
        probs = torch.distributions.Categorical(logits=logits)
        actions = probs.sample()
        return actions, probs.log_prob(actions)



class PolicyGNN_2(torch.nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        self.conv1 = GCNConv(n_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.out = torch.nn.Linear(32, n_actions)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        logits = self.out(x)     # [n_APs, n_actions]
        return logits




from gymnasium import ObservationWrapper

class FlattenObs(ObservationWrapper):
    def observation(self, observation):
        H_flat = observation["H"].flatten()
        mu = observation["mu"]
        return np.concatenate([H_flat, mu])
