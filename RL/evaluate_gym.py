import torch
import numpy as np
from network_environment import NetworkEnviroment
from gnn import GNN
from utils import get_gnn_inputs

class GNNAgent:
    
    def predict_with_probs(self, observation):
        """Versión que devuelve tanto acciones como probabilidades"""
        channel_matrix = torch.from_numpy(observation).float().unsqueeze(0)
        x_tensor = torch.zeros((1, self.num_links, 1))
        
        graph_data_list = get_gnn_inputs(x_tensor, channel_matrix)
        graph_data = graph_data_list[0]
        
        with torch.no_grad():
            psi = self.gnn_model.forward(
                graph_data.x, 
                graph_data.edge_index, 
                graph_data.edge_attr
            )
            psi = psi.view(1, self.num_links, -1)
            probs = torch.softmax(psi, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample().squeeze(0)
            
        return actions.numpy(), probs.numpy()

def evaluate_gym(model_path, num_links, num_channels, num_layers, K, num_episodes=10):
    """Evaluar el modelo entrenado"""
    
    # Configurar entorno
    env = NetworkEnviroment(
        num_links=num_links,
        num_channels=num_channels,
        num_layers=num_layers,
        K=K,
        batch_size=1,
        epochs=num_episodes
    )
    
    # Cargar modelo
    num_actions = 1 + num_channels * 2  # 2 power levels
    gnn_model = GNN(1, 1, num_actions, num_layers, False, K)
    gnn_model.load_state_dict(torch.load(model_path))
    gnn_model.eval()
    
    agent = GNNAgent(gnn_model, num_links, num_actions)
    
    total_rewards = []
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, probs = agent.predict_with_probs(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episodio {episode}: Recompensa = {episode_reward:.4f}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Recompensa promedio en {num_episodes} episodios: {avg_reward:.4f}")
    
    return avg_reward

if __name__ == '__main__':
    # Ejemplo de uso
    evaluate_gym(
        model_path='ruta/a/tu/modelo.pth',
        num_links=20,
        num_channels=11,
        num_layers=5,
        K=3,
        num_episodes=10
    )