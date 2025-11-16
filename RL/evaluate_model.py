"""
Script para evaluar modelos ya entrenados
"""
import numpy as np
import torch
from stable_baselines3 import PPO
from network_env import NetworkEnvironment
from utils import load_channel_matrix
import matplotlib.pyplot as plt


def evaluate_trained_model(model_path, n_episodes=20, render=False):
    """
    Carga y evalúa un modelo entrenado
    
    Args:
        model_path: Ruta al modelo guardado (.zip)
        n_episodes: Número de episodios para evaluar
        render: Si True, muestra información de cada paso
    """
    print("="*60)
    print(f"  EVALUACIÓN DE MODELO: {model_path}")
    print("="*60)
    
    # Configuración del entorno (debe coincidir con el entrenamiento)
    num_links = 5
    num_channels = 3
    num_power_levels = 2
    max_steps = 100
    
    print("\n📡 Cargando datos...")
    channel_matrix_iter = load_channel_matrix(
        building_id=990,
        b5g=False,
        num_links=num_links,
        synthetic=False,
        shuffle=True,
        repeat=True
    )
    
    print("🏗️  Creando entorno...")
    env = NetworkEnvironment(
        num_links=num_links,
        num_channels=num_channels,
        num_power_levels=num_power_levels,
        max_steps=max_steps,
        eps=5e-4,
        max_antenna_power_dbm=6,
        sigma=1e-4,
        device="cpu",
        channel_matrix_iter=channel_matrix_iter
    )
    
    print(f"📦 Cargando modelo desde {model_path}...")
    model = PPO.load(model_path)
    print("✓ Modelo cargado")
    
    # Métricas de evaluación
    all_rewards = []
    all_rates = []
    all_power_violations = []
    all_steps = []
    episode_data = []
    
    print(f"\n🎯 Evaluando por {n_episodes} episodios...\n")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_rates = []
        episode_violations = []
        episode_mu = []
        step_count = 0
        
        while step_count < max_steps:
            # Usar política directamente
            with torch.no_grad():
                obs_tensor = {
                    'channel_matrix': torch.FloatTensor(obs['channel_matrix']).unsqueeze(0),
                    'mu_k': torch.FloatTensor(obs['mu_k']).unsqueeze(0)
                }
                actions, _, _ = model.policy.forward(obs_tensor, deterministic=True)
                action = actions.cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_rates.append(info.get('rate', 0))
            episode_violations.append(info.get('power_constraint', 0))
            episode_mu.append(info.get('avg_mu', 0))
            step_count += 1
            
            if render and step_count % 10 == 0:
                print(f"  Step {step_count}: Rate={info.get('rate', 0):.4f}, "
                      f"Violation={info.get('power_constraint', 0):.4f}")
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_rates.append(np.mean(episode_rates) if episode_rates else 0)
        all_power_violations.append(np.mean(episode_violations) if episode_violations else 0)
        all_steps.append(step_count)
        
        episode_data.append({
            'rates': episode_rates,
            'violations': episode_violations,
            'mu': episode_mu
        })
        
        print(f"Episode {episode+1:2d}/{n_episodes}: "
              f"Reward={episode_reward:7.2f}, "
              f"Avg Rate={np.mean(episode_rates):6.4f}, "
              f"Avg Violation={np.mean(episode_violations):7.4f}, "
              f"Steps={step_count:3d}")
    
    # Estadísticas finales
    print("\n" + "="*60)
    print("  RESULTADOS FINALES")
    print("="*60)
    print(f"Reward promedio:      {np.mean(all_rewards):8.2f} ± {np.std(all_rewards):.2f}")
    print(f"Rate promedio:        {np.mean(all_rates):8.4f} ± {np.std(all_rates):.4f}")
    print(f"Violación promedio:   {np.mean(all_power_violations):8.4f} ± {np.std(all_power_violations):.4f}")
    print(f"Steps promedio:       {np.mean(all_steps):8.1f} ± {np.std(all_steps):.1f}")
    print(f"Reward máximo:        {np.max(all_rewards):8.2f}")
    print(f"Reward mínimo:        {np.min(all_rewards):8.2f}")
    print("="*60)
    
    # Visualizaciones
    plot_evaluation_results(all_rewards, all_rates, all_power_violations, episode_data)
    
    env.close()
    
    return {
        'rewards': all_rewards,
        'rates': all_rates,
        'violations': all_power_violations,
        'steps': all_steps,
        'episode_data': episode_data
    }


def plot_evaluation_results(rewards, rates, violations, episode_data):
    """Crea visualizaciones de los resultados de evaluación"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Rewards por episodio
    axes[0, 0].plot(rewards, marker='o', linewidth=2, markersize=4)
    axes[0, 0].axhline(np.mean(rewards), color='r', linestyle='--', 
                       label=f'Media: {np.mean(rewards):.2f}')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Reward Total')
    axes[0, 0].set_title('Reward por Episodio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Rates promedio por episodio
    axes[0, 1].plot(rates, marker='s', color='green', linewidth=2, markersize=4)
    axes[0, 1].axhline(np.mean(rates), color='r', linestyle='--',
                       label=f'Media: {np.mean(rates):.4f}')
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('Rate Promedio')
    axes[0, 1].set_title('Rate Promedio por Episodio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Violaciones promedio por episodio
    axes[1, 0].plot(violations, marker='^', color='orange', linewidth=2, markersize=4)
    axes[1, 0].axhline(0, color='g', linestyle='--', label='Sin violación')
    axes[1, 0].axhline(np.mean(violations), color='r', linestyle='--',
                       label=f'Media: {np.mean(violations):.4f}')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Violación de Potencia')
    axes[1, 0].set_title('Violación de Restricción de Potencia')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Evolución del rate en un episodio ejemplo
    example_ep = len(episode_data) // 2  # Episodio del medio
    axes[1, 1].plot(episode_data[example_ep]['rates'], linewidth=2)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].set_title(f'Evolución del Rate (Episodio {example_ep+1})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gráficos guardados en: evaluation_results.png")
    plt.show()


def compare_with_baseline(model_path, n_episodes=10):
    """Compara el modelo entrenado con una política aleatoria"""
    
    print("\n" + "="*60)
    print("  COMPARACIÓN CON BASELINE ALEATORIO")
    print("="*60)
    
    # Evaluar modelo entrenado
    print("\n1️⃣  Evaluando modelo GNN-PPO...")
    trained_results = evaluate_trained_model(model_path, n_episodes, render=False)
    
    # Evaluar baseline aleatorio
    print("\n2️⃣  Evaluando baseline aleatorio...")
    
    num_links = 5
    num_channels = 3
    num_power_levels = 2
    max_steps = 100
    
    channel_matrix_iter = load_channel_matrix(
        building_id=990,
        b5g=False,
        num_links=num_links,
        synthetic=False,
        shuffle=True,
        repeat=True
    )
    
    env = NetworkEnvironment(
        num_links=num_links,
        num_channels=num_channels,
        num_power_levels=num_power_levels,
        max_steps=max_steps,
        eps=5e-4,
        max_antenna_power_dbm=6,
        sigma=1e-4,
        device="cpu",
        channel_matrix_iter=channel_matrix_iter
    )
    
    random_rewards = []
    random_rates = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_rates = []
        
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_rates.append(info.get('rate', 0))
            
            if terminated or truncated:
                break
        
        random_rewards.append(episode_reward)
        random_rates.append(np.mean(episode_rates) if episode_rates else 0)
    
    env.close()
    
    # Comparación
    print("\n" + "="*60)
    print("  RESULTADOS DE LA COMPARACIÓN")
    print("="*60)
    print(f"{'Métrica':<25} {'GNN-PPO':>12} {'Aleatorio':>12} {'Mejora':>12}")
    print("-"*60)
    
    trained_reward = np.mean(trained_results['rewards'])
    random_reward = np.mean(random_rewards)
    improvement_reward = ((trained_reward - random_reward) / abs(random_reward)) * 100
    
    trained_rate = np.mean(trained_results['rates'])
    random_rate = np.mean(random_rates)
    improvement_rate = ((trained_rate - random_rate) / abs(random_rate)) * 100
    
    print(f"{'Reward promedio':<25} {trained_reward:>12.2f} {random_reward:>12.2f} {improvement_reward:>11.1f}%")
    print(f"{'Rate promedio':<25} {trained_rate:>12.4f} {random_rate:>12.4f} {improvement_rate:>11.1f}%")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./models/gnn_ppo_network_final.zip"
    
    print("\n🚀 Iniciando evaluación...\n")
    
    # Evaluación estándar
    results = evaluate_trained_model(model_path, n_episodes=20, render=False)
    
    # Comparación con baseline
    compare_with_baseline(model_path, n_episodes=10)
    
    print("\n✅ Evaluación completada")