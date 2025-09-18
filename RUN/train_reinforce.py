import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from gnn_env import GNNEnv
from gnn import GNN
from torch_geometric.data import Batch

# Hyperparámetros
num_links = 5
num_channels = 3
power_levels = 2
num_layers = 4
hidden_dim = 16
output_dim = (num_channels * power_levels + 1)
epochs = 200
learning_rate = 1e-3
gamma = 0.99  # Factor de descuento para REINFORCE

def reinforce_train():
    # Inicializa el entorno y el modelo
    env = GNNEnv(num_links=num_links, num_channels=num_channels, power_levels=power_levels)
    
    # La GNN ahora predice la probabilidad de 7 acciones para cada enlace
    agent = GNN(
        input_dim=env.observation_space['x'].shape[1],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.1,
        K=3
    )
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    all_rewards = []
    
    for epoch in range(epochs):
        # Reinicia el entorno para cada episodio
        observation, _ = env.reset()
        
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            # Prepara los datos para la GNN (agregación en un solo lote)
            # Nota: Esto es necesario porque el agente espera un batch
            data_batch = Batch.from_data_list([env.current_data])
            x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
            
            # Forward pass para obtener las log-probabilidades de las acciones
            output = agent.forward(x, edge_index, edge_attr)
            
            # Redimensiona la salida para que coincida con el espacio de acciones
            # output tiene forma [num_links, output_dim]
            output = output.view(1, num_links, output_dim)
            
            # Obtiene las probabilidades de cada acción
            probs = F.softmax(output, dim=-1).squeeze(0)
            
            # Crea una distribución categórica para muestrear acciones
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            # Guarda el log-prob de la acción tomada
            log_prob = m.log_prob(action)
            
            # Toma un paso en el entorno
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

        # Cálculo de la pérdida de REINFORCE
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        # Normalización de las recompensas para reducir la varianza
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            loss.append(-log_prob.mean() * R)
        
        loss = torch.stack(loss).sum()
        
        # Actualización de la red
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_rewards.append(np.sum(rewards))

        if epoch % 10 == 0:
            print(f"Época {epoch}: Recompensa total = {all_rewards[-1]:.4f}")

    print("Entrenamiento con REINFORCE finalizado.")
    
    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.plot(all_rewards)
    plt.title('Recompensa Total por Época')
    plt.xlabel('Época')
    plt.ylabel('Recompensa Total')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    reinforce_train()