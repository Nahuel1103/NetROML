from torch_geometric.data import Data

def obs_to_graph(obs):
    H = obs["H"]            # [n_APs, n_APs]
    mu = obs["mu"]          # [n_APs]
    n_APs = H.shape[0]

    # Nodos: cada AP
    x = torch.stack([mu, torch.ones_like(mu) * Pmax], dim=1)  # [n_APs, 2]
    # x_i = [mu_i, Pmax_i]  ← features del nodo i

    # Aristas: derivadas de H (interferencia)
    src, dst = torch.nonzero(H > 0, as_tuple=True)
    edge_weight = H[src, dst]

    return Data(x=x, edge_index=torch.stack([src, dst]), edge_attr=edge_weight)
