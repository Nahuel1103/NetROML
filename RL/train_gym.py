"""
Script de entrenamiento para NetworkEnvironment con GNN Policy
"""
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from gnn_sb3_policy import GNNActorCriticPolicy
from network_env import NetworkEnvironment
from utils import load_channel_matrix


def train_network_env():
    """Entrena el agente GNN-PPO en NetworkEnvironment"""
    
    # Configuración del entorno
    num_links = 5
    num_channels = 3
    num_power_levels = 2
    max_steps = 100
    
    print("🔧 Configurando entorno...")
    print(f"  - Enlaces: {num_links}")
    print(f"  - Canales: {num_channels}")
    print(f"  - Niveles de potencia: {num_power_levels}")
    
    # Cargar iterador de matrices de canal
    print("\n📡 Cargando matrices de canal...")
    channel_matrix_iter = load_channel_matrix(
        building_id=990,
        b5g=False,  # 2.4 GHz
        num_links=num_links,
        synthetic=False,  # True para datos sintéticos
        shuffle=True,
        repeat=True,  # Reinicia automáticamente
        train=True
    )
    
    # Crear entorno
    env = NetworkEnvironment(
        num_links=num_links,
        num_channels=num_channels,
        num_power_levels=num_power_levels,
        max_steps=max_steps,
        eps=5e-5,
        max_antenna_power_dbm=6,
        sigma=1e-4,
        device="auto",
        channel_matrix_iter=channel_matrix_iter
    )

    # wrapped_env = env.wrappers.RecordEpisodeStatistics(env, 50)
    
    # Verificar entorno
    print("\n✓ Verificando entorno...")
    try:
        check_env(env, warn=True, skip_render_check=True)
        print("  ✓ Entorno válido")
    except Exception as e:
        print(f"  ⚠️  Error en verificación: {e}")
        print("  Continuando de todas formas...")
    
    # Test manual del entorno
    print("\n🧪 Test del entorno...")
    obs, info = env.reset()
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Obs keys: {obs.keys()}")
    print(f"  - Channel matrix shape: {obs['channel_matrix'].shape}")
    print(f"  - Mu_k shape: {obs['mu_k'].shape}")
    
    # Test de un paso
    action = env.action_space.sample()
    print(f"  - Action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  - Reward: {reward:.4f}")
    print(f"  - Info: {info}")
    print("  ✓ Test completado\n")
    
    # Resetear para entrenamiento
    env.reset()
    
    # Configuración de la política GNN
    policy_kwargs = dict(
        gnn_hidden_dim=32,
        gnn_num_layers=3,
        K=3
    )
    
    print("🤖 Creando modelo PPO con GNN...")
    print(f"  - Hidden dim: {policy_kwargs['gnn_hidden_dim']}")
    print(f"  - GNN layers: {policy_kwargs['gnn_num_layers']}")
    print(f"  - K-hop: {policy_kwargs['K']}")
    
    # Crear modelo PPO
    model = PPO(
        GNNActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,      # Horizonte de recolección
        batch_size=64,    # Tamaño de batch para actualización
        n_epochs=10,       # Épocas de actualización
        gamma=0.99,        # Factor de descuento
        gae_lambda=0.95,   # GAE lambda
        clip_range=0.2,    # PPO clip range
        ent_coef=0.01,     # Coeficiente de entropía (exploración)
        vf_coef=0.5,       # Coeficiente de value function
        max_grad_norm=0.5, # Gradient clipping
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Callbacks para guardar checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path="./checkpoints/",
        name_prefix="gnn_ppo_network"
    )
    
    print("\n🚀 Iniciando entrenamiento...")
    print("="*60)
    
    # Entrenar
    total_timesteps = 10000
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="gnn_ppo_network",
        progress_bar=True
    )
    
    print("\n✓ Entrenamiento completado")
    
    # Guardar modelo final
    model_path = "./models/gnn_ppo_network_final"
    model.save(model_path)
    print(f"✓ Modelo guardado en: {model_path}")
    
    # Evaluar modelo entrenado
    evaluate_model(model, n_episodes=10)
    
    return model, env


def evaluate_model(model, n_episodes=10):
    """
    Evalúa el modelo entrenado creando un entorno limpio
    """
    from utils import load_channel_matrix
    
    print(f"\n📊 Evaluando modelo por {n_episodes} episodios...")
    
    # ✅ Crear entorno de evaluación limpio
    num_links = 5
    num_channels = 3
    num_power_levels = 2
    max_steps = 100
    
    channel_matrix_iter = load_channel_matrix(
        building_id=990,
        b5g=False,  # 2.4 GHz
        num_links=num_links,
        synthetic=False,  # True para datos sintéticos
        shuffle=True,
        repeat=True,  # Reinicia automáticamente
        train=False  # USAR VALIDACIÓN
    )
    
    # Crear entorno
    eval_env = NetworkEnvironment(
        num_links=num_links,
        num_channels=num_channels,
        num_power_levels=num_power_levels,
        max_steps=max_steps,
        eps=5e-3,
        max_antenna_power_dbm=6,
        sigma=1e-4,
        device="cpu",
        channel_matrix_iter=channel_matrix_iter
    )
    
    total_rewards = []
    total_rates = []
    total_power_violations = []
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_rates = []
        episode_violations = []
        step_count = 0
        
        while step_count < max_steps:
            # ✅ Usar método interno de la política para evitar problemas con VecEnv
            with torch.no_grad():
                # Convertir obs a tensors
                obs_tensor = {
                    'channel_matrix': torch.FloatTensor(obs['channel_matrix']).unsqueeze(0),
                    'mu_k': torch.FloatTensor(obs['mu_k']).unsqueeze(0)
                }
                
                # Obtener acción de la política directamente
                actions, _, _ = model.policy.forward(obs_tensor, deterministic=True)
                action = actions.cpu().numpy()[0]  # Remover batch dimension
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            episode_reward += reward
            episode_rates.append(info.get('rate', 0))
            episode_violations.append(info.get('power_constraint', 0))
            step_count += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_rates.append(np.mean(episode_rates) if episode_rates else 0)
        total_power_violations.append(np.mean(episode_violations) if episode_violations else 0)
        
        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Avg Rate={np.mean(episode_rates) if episode_rates else 0:.4f}, "
              f"Avg Violation={np.mean(episode_violations) if episode_violations else 0:.4f}, "
              f"Steps={step_count}")
    
    print(f"\n📈 Resultados de evaluación ({n_episodes} episodios):")
    print(f"  - Reward promedio: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  - Rate promedio: {np.mean(total_rates):.4f} ± {np.std(total_rates):.4f}")
    print(f"  - Violación promedio: {np.mean(total_power_violations):.4f}")
    
    eval_env.close()
    return total_rewards, total_rates, total_power_violations


if __name__ == "__main__":
    print("="*60)
    print("  Entrenamiento GNN-PPO para Optimización de Redes Wi-Fi")
    print("="*60 + "\n")
    
    model, env = train_network_env()
    
    print("\n" + "="*60)
    print("  ✓ Proceso completado exitosamente")
    print("="*60)