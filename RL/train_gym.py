"""
Sistema de Entrenamiento para Políticas de Asignación de Potencia
Utiliza Redes Neuronales de Grafos y aprendizaje por refuerzo.
"""

import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
from network_environment import NetworkEnvironment
from gnn import GNN
from plot_results_torch import plot_results


class GNNAgent:
    """
    Agente para toma de decisiones en redes inalámbricas.
    
    Utiliza una GNN para mapear estados de la red a políticas
    de asignación de potencia óptimas.
    """

    def __init__(self, gnn_model, num_links, num_actions):
        self.gnn_model = gnn_model
        self.num_links = num_links
        self.num_actions = num_actions
        
    def predict(self, graph_data):
        """
        Genera decisiones de asignación de potencia basadas en el estado actual.
        
        Returns:
            actions: Vector de decisiones discretas por enlace
            action_probs: Distribución de probabilidad de las decisiones
        """
        with torch.no_grad():
            psi = self.gnn_model.forward(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            psi = psi.view(1, self.num_links, -1)
            probs = torch.softmax(psi, dim=-1)
            
            # Estabilización numérica
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            if not torch.all(torch.isfinite(probs)):
                probs = torch.where(torch.isfinite(probs), probs, torch.ones_like(probs) / probs.size(-1))
            
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample().squeeze(0)
            
        return actions.numpy(), probs.squeeze(0).cpu().numpy()


def train_gym(building_id=990, b5g=0, num_links=5, num_channels=3, num_power_levels=2, 
              num_layers=5, K=3, batch_size=64, epochs=100, eps=5e-4, mu_lr=1e-4, 
              synthetic=0, max_antenna_power_dbm=6, sigma=1e-4):
    """
    Loop principal de entrenamiento para políticas de asignación de potencia.
    
    Combina optimización Lagrangiana con gradientes de política para
    entrenar redes neuronales en problemas con restricciones.
    """
    
    # Configuración de reproducibilidad
    rn = 267309
    rn1 = 502321
    torch.manual_seed(rn)
    np.random.seed(rn1)
    
    print("Inicializando entorno de red Wi-Fi...")
    
    env = NetworkEnvironment(
        building_id=building_id,
        b5g=b5g,
        num_links=num_links,
        num_channels=num_channels,
        num_power_levels=num_power_levels,
        num_layers=num_layers,
        K=K,
        batch_size=batch_size,
        epochs=epochs,
        eps=eps,
        mu_lr=mu_lr,
        synthetic=synthetic,
        max_antenna_power_dbm=max_antenna_power_dbm,
        sigma=sigma
    )
    
    num_actions = 1 + num_channels * num_power_levels
    
    print("Configuración del sistema:")
    print(f"  Enlaces: {num_links}, Canales: {num_channels}")
    print(f"  Espacio de acción: {num_actions} decisiones por enlace")
    print(f"  Observaciones: {env.observation_space.shape}")

    # Inicialización del modelo
    input_dim = 1
    hidden_dim = 1
    output_dim = num_actions
    dropout = False
    
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)

    agent = GNNAgent(gnn_model, num_links, num_actions)

    # Sistema de métricas
    objective_function_values = []
    power_constraint_values = []  
    loss_values = []
    mu_k_values = []
    probs_values = []

    print("Iniciando proceso de entrenamiento...")

    for episode in range(epochs):
        print(f"Procesando episodio {episode}")
        
        observation, info = env.reset()
        graph_data = info["graph_data"]
        done = False
        episode_probs = []
        episode_power_constraints = []
        episode_objectives = []
        episode_mu_k = []

        while not done:
            action, probs = agent.predict(graph_data)
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Recolección de métricas
            episode_probs.append(probs)
            episode_power_constraints.append(info["power_constraint"])
            episode_objectives.append(-reward)
            episode_mu_k.append(info["mu_k"].copy())
            
            graph_data = info["graph_data"]

        # Consolidación de estadísticas del episodio
        if episode % 10 == 0:
            avg_probs = np.mean(episode_probs, axis=0)
            avg_power_constraint = np.mean(episode_power_constraints)
            avg_objective = np.mean(episode_objectives)
            avg_mu_k = np.mean(episode_mu_k, axis=0)
            
            probs_values.append(avg_probs)
            power_constraint_values.append(avg_power_constraint)
            objective_function_values.append(avg_objective)
            mu_k_values.append(avg_mu_k)
            
            print(f"Episodio {episode} | "
                  f"Objetivo: {avg_objective:.4f} | "
                  f"Restricción: {avg_power_constraint:.4f}")

    # Persistencia de resultados
    print("Finalizando entrenamiento y guardando resultados...")
    
    path = plot_results(
        building_id=building_id,
        b5g=b5g,
        normalized_psi=torch.tensor(probs_values),
        normalized_psi_values=probs_values,
        num_layers=num_layers,
        K=K,
        batch_size=batch_size,
        epochs=epochs,
        rn=rn,
        rn1=rn1,
        eps=eps,
        mu_lr=mu_lr,
        objective_function_values=objective_function_values,
        power_constraint_values=power_constraint_values,
        loss_values=loss_values,
        mu_k_values=mu_k_values,
        train=True
    )

    # Almacenamiento de artefactos
    metrics_file = path + 'training_metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(objective_function_values, f)

    model_path = path + 'trained_gnn_weights.pth'
    torch.save(gnn_model.state_dict(), model_path)

    print(f"Modelo y métricas guardados en: {path}")
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenamiento de políticas para redes inalámbricas'
    )
    
    parser.add_argument('--building_id', type=int, default=990)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=5)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_power_levels', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=5e-4)
    parser.add_argument('--mu_lr', type=float, default=5e-4)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--max_antenna_power_dbm', type=int, default=6)
    
    args = parser.parse_args()
    
    print("Configuración de entrenamiento:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    train_gym(
        building_id=args.building_id,
        b5g=args.b5g,
        num_links=args.num_links,
        num_channels=args.num_channels,
        num_power_levels=args.num_power_levels,
        num_layers=args.num_layers,
        K=args.k,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eps=args.eps,
        mu_lr=args.mu_lr,
        synthetic=args.synthetic,
        max_antenna_power_dbm=args.max_antenna_power_dbm
    )