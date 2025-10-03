"""
Sistema de Entrenamiento para Políticas de Asignación de Potencia (CORREGIDO)
Utiliza Redes Neuronales de Grafos y aprendizaje por refuerzo con REINFORCE.
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
        
    def predict(self, graph_data, training=False):
        """
        Genera decisiones de asignación de potencia basadas en el estado actual.
        
        Args:
            graph_data: Datos del grafo de entrada
            training: Si True, mantiene el grafo computacional para backprop
        
        Returns:
            actions: Vector de decisiones discretas por enlace
            action_probs: Distribución de probabilidad de las decisiones
            log_probs: Log-probabilidades para REINFORCE (solo si training=True)
        """
        if training:
            # Modo entrenamiento: mantener gradientes
            psi = self.gnn_model.forward(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            
            # DEBUG: Verificar dimensiones
            batch_size = graph_data.x.size(0) // self.num_links
            
            # Reshape: [batch_size * num_links, output_dim] -> [batch_size, num_links, num_actions]
            psi = psi.view(batch_size, self.num_links, self.num_actions)
            
            # Tomar solo el primer elemento del batch
            psi = psi[0:1]  # Shape: [1, num_links, num_actions]
            
            probs = torch.softmax(psi, dim=-1)
            
            # Estabilización numérica
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Verificar que probs tenga la forma correcta
            assert probs.shape == (1, self.num_links, self.num_actions), \
                f"Shape incorrecto: {probs.shape}, esperado (1, {self.num_links}, {self.num_actions})"
            
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample().squeeze(0)
            log_probs = dist.log_prob(actions)
            
            # Validar que las acciones estén en el rango correcto
            assert torch.all(actions >= 0) and torch.all(actions < self.num_actions), \
                f"Acciones fuera de rango: {actions}, min={actions.min()}, max={actions.max()}"
            
            return actions, probs.squeeze(0), log_probs
        else:
            # Modo inferencia: sin gradientes
            with torch.no_grad():
                psi = self.gnn_model.forward(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
                
                batch_size = graph_data.x.size(0) // self.num_links
                psi = psi.view(batch_size, self.num_links, self.num_actions)
                psi = psi[0:1]
                
                probs = torch.softmax(psi, dim=-1)
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                dist = torch.distributions.Categorical(probs=probs)
                actions = dist.sample().squeeze(0)
                
            return actions.numpy(), probs.squeeze(0).cpu().numpy(), None


def train_gym(building_id=990, b5g=0, num_links=5, num_channels=3, num_power_levels=2, 
              num_layers=5, K=3, batch_size=64, epochs=100, eps=5e-4, mu_lr=1e-4, 
              synthetic=0, max_antenna_power_dbm=6, sigma=1e-4):
    """
    Loop principal de entrenamiento para políticas de asignación de potencia.
    
    Combina optimización Lagrangiana con REINFORCE para entrenar redes neuronales 
    en problemas con restricciones.
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
    hidden_dim = 16  # Aumentado para mejor capacidad
    output_dim = num_actions
    dropout = False
    
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)
    
    # DEBUG: Verificar la salida de la GNN con datos de prueba
    print("\n=== DEBUG: Verificando arquitectura GNN ===")
    observation, info = env.reset()
    test_graph = info["graph_data"]
    with torch.no_grad():
        test_output = gnn_model.forward(test_graph.x, test_graph.edge_index, test_graph.edge_attr)
        print(f"Input shape: x={test_graph.x.shape}, edge_index={test_graph.edge_index.shape}")
        print(f"Output shape de GNN: {test_output.shape}")
        print(f"Esperado: [{batch_size * num_links}, {num_actions}]")
        print(f"Output_dim configurado: {output_dim}")
        print(f"Primeros valores de salida: {test_output[:5, :].flatten()[:10]}")
    print("=" * 50 + "\n")
    
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)

    agent = GNNAgent(gnn_model, num_links, num_actions)

    # Sistema de métricas
    objective_function_values = []
    power_constraint_values = []  
    loss_values = []
    mu_k_values = []
    probs_values = []

    print("Iniciando proceso de entrenamiento con REINFORCE...")

    for episode in range(epochs):
        observation, info = env.reset()
        graph_data = info["graph_data"]
        done = False
        
        # Buffers para REINFORCE
        episode_log_probs = []
        episode_rewards = []
        episode_probs = []
        episode_power_constraints = []
        episode_objectives = []
        episode_mu_k = []

        # Recolección de trayectoria
        while not done:
            # FIX: Modo entrenamiento para mantener gradientes
            action_tensor, probs, log_probs = agent.predict(graph_data, training=True)
            action = action_tensor.detach().cpu().numpy()
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Almacenar para REINFORCE
            episode_log_probs.append(log_probs)
            episode_rewards.append(reward)
            
            # Métricas
            episode_probs.append(probs.detach().cpu().numpy())
            episode_power_constraints.append(info["power_constraint"])
            episode_objectives.append(-reward if np.isfinite(reward) else 0.0)
            episode_mu_k.append(info["mu_k"].copy())
            
            graph_data = info["graph_data"]

        # Actualización de política con REINFORCE
        if len(episode_log_probs) > 0:
            # Calcular retornos (rewards acumulados)
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                if np.isfinite(r):
                    R = r + 0.99 * R  # gamma = 0.99
                    returns.insert(0, R)
                else:
                    returns.insert(0, 0.0)
            
            # Normalizar retornos para estabilidad
            returns = torch.tensor(returns, dtype=torch.float32)
            if len(returns) > 1 and returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calcular pérdida de política
            policy_loss = []
            for log_prob, R in zip(episode_log_probs, returns):
                # REINFORCE: -log_prob * return
                policy_loss.append(-log_prob.sum() * R)
            
            if len(policy_loss) > 0:
                policy_loss = torch.stack(policy_loss).sum()
                
                # Backpropagation
                optimizer.zero_grad()
                policy_loss.backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                loss_values.append(policy_loss.item())

        # Consolidación de estadísticas del episodio
        if episode % 10 == 0:
            avg_probs = np.mean(episode_probs, axis=0)
            avg_power_constraint = np.mean(episode_power_constraints)
            avg_objective = np.mean(episode_objectives)
            avg_mu_k = np.mean(episode_mu_k, axis=0)
            avg_loss = np.mean(loss_values[-10:]) if loss_values else 0.0
            
            probs_values.append(avg_probs)
            power_constraint_values.append(avg_power_constraint)
            objective_function_values.append(avg_objective)
            mu_k_values.append(avg_mu_k)
            
            print(f"Episodio {episode:3d} | "
                  f"Objetivo: {avg_objective:8.4f} | "
                  f"Restricción: {avg_power_constraint:8.4f} | "
                  f"Loss: {avg_loss:8.4f}")

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
        pickle.dump({
            'objective_function_values': objective_function_values,
            'power_constraint_values': power_constraint_values,
            'loss_values': loss_values,
            'mu_k_values': mu_k_values
        }, f)

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