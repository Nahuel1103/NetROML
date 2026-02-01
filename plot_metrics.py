import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import networkx as nx
from pathlib import Path
from torch_geometric.utils import to_networkx

def plot_training_metrics(csv_file='training_metrics.csv', output_dir='plots'):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    sns.set_theme(style="whitegrid")
    
    # 1. Total Reward Sum (Return)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='episode', y='reward_sum', label='Episode Return')
    # Rolling average
    if len(df) > 10:
        df['reward_rolling'] = df['reward_sum'].rolling(window=10).mean()
        sns.lineplot(data=df, x='episode', y='reward_rolling', label='Rolling Avg (10)', color='orange')
    plt.title('Training Return (Reward Sum)')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.savefig(os.path.join(output_dir, 'training_return.png'))
    plt.close()
    
    # 2. Total Rate (Throughput)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='episode', y='total_rate_mbps', color='green')
    plt.title('Network Throughput')
    plt.xlabel('Episode')
    plt.ylabel('Total Rate (Mbps)')
    plt.savefig(os.path.join(output_dir, 'throughput.png'))
    plt.close()

    # 3. Fairness
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='episode', y='mean_fairness', color='purple')
    plt.title("Jain's Fairness Index")
    plt.xlabel('Episode')
    plt.ylabel('Fairness Index (0-1)')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'fairness.png'))
    plt.close()
    
    # 4. Loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='episode', y='loss', color='red')
    plt.title('Policy Loss')
    plt.xlabel('Episode')
    plt.yscale('symlog') # Handle negative/positive transitions if any, though loss usually pos
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    # 5. Learning Rate & Grad Norm
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Learning Rate', color=color)
    ax1.plot(df['episode'], df['lr'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:gray'
    ax2.set_ylabel('Gradient Norm', color=color)  
    ax2.plot(df['episode'], df['grad_norm'], color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Learning Dynamics')
    plt.savefig(os.path.join(output_dir, 'learning_dynamics.png'))
    plt.close()
    
    print(f"Training plots saved to {output_dir}/")


def plot_snapshot_topology(snapshot_path, output_dir='plots/snapshots'):
    """
    Visualiza la topología (conexiones) de un snapshot guardado.
    """
    if not os.path.exists(snapshot_path):
        return

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        snapshot = torch.load(snapshot_path, weights_only=False)
    except Exception as e:
        print(f"Error loading snapshot {snapshot_path}: {e}")
        return

    obs = snapshot['obs']
    episode = snapshot['episode']
    
    # Extraer aristas Uplink (Client -> AP) que indican conexión activa
    # obs['client', 'connected_to', 'ap'].edge_index
    edge_index = obs['client', 'connected_to', 'ap'].edge_index
    
    if edge_index.shape[1] == 0:
        print(f"Snapshot Ep {episode}: No active connections to plot.")
        return

    # Convertir a Grafo NetworkX para plotear
    # Nodos: APs (0..N-1) y Clientes (N..N+M-1)
    # Pero HeteroData es complejo. Hacemos grafo simple manual.
    G = nx.Graph()
    
    num_aps = obs['ap'].num_nodes
    num_clients = obs['client'].num_nodes
    
    # APs
    pos_ap = {}
    for i in range(num_aps):
        ap_node_id = f"AP_{i}"
        G.add_node(ap_node_id, color='red', type='ap')
        # Distribución circular o fija para APs?
        # Simular grid simple o circulo
        import math
        angle = 2 * math.pi * i / num_aps
        pos_ap[ap_node_id] = (math.cos(angle)*2, math.sin(angle)*2)

    # Clients
    # Posicionar orbitando a su AP conectado (o random si no)
    pos_client = {}
    
    # Aristas
    client_indices = edge_index[0].numpy()
    ap_indices = edge_index[1].numpy()
    
    for c_idx, a_idx in zip(client_indices, ap_indices):
        c_node_id = f"C_{c_idx}"
        a_node_id = f"AP_{a_idx}"
        G.add_edge(c_node_id, a_node_id)
        
        # Posición relativa al AP con ruido
        import numpy as np
        ap_pos = pos_ap[a_node_id]
        noise = np.random.normal(0, 0.3, 2)
        pos_client[c_node_id] = (ap_pos[0] + noise[0], ap_pos[1] + noise[1])
        
        G.add_node(c_node_id, color='blue', type='client')

    # Combined Layout
    pos = {**pos_ap, **pos_client}
    
    plt.figure(figsize=(10, 10))
    
    # Draw APs
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n,d in G.nodes(data=True) if d.get('type')=='ap'], 
                           node_color='red', node_size=500, label='APs')
    # Draw Clients
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n,d in G.nodes(data=True) if d.get('type')=='client'], 
                           node_color='blue', node_size=100, label='Clients')
    
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels={n:n for n,d in G.nodes(data=True) if d.get('type')=='ap'}, font_color='white')

    plt.title(f"Network Topology - Episode {episode}")
    plt.legend()
    plt.axis('off')
    
    filename = f"topology_ep_{episode}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Snapshot plot saved: {os.path.join(output_dir, filename)}")

def process_all_snapshots(models_dir='saved_models'):
    p = Path(models_dir)
    if not p.exists():
        return
        
    for f in p.glob('snapshot_ep_*.pt'):
        plot_snapshot_topology(str(f))

if __name__ == "__main__":
    plot_training_metrics()
    process_all_snapshots()
