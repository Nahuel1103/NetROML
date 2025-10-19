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
