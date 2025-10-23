"""
Script de entrenamiento usando Stable-Baselines3 con tu environment
"""
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs import APNetworkEnv
from custom_policy import GNNActorCriticPolicy


def make_channel_iterator(channel_matrices):
    """
    Crea un iterador de matrices de canal
    """
    for H in channel_matrices:
        yield H


def load_channel_matrices(building_id=990, b5g=False, train=True):
    """
    Carga las matrices de canal desde tus datos
    Similar a lo que hacías en graphs_to_tensor
    """
    # TODO: Implementar carga de datos real
    # from utils import graphs_to_tensor, get_gnn_inputs
    
    # Por ahora, matrices sintéticas
    n_samples = 7000 if train else 1000
    n_APs = 5
    
    channel_matrices = []
    for _ in range(n_samples):
        # Matriz de canal aleatoria
        H = np.random.rand(n_APs, n_APs).astype(np.float32)
        np.fill_diagonal(H, H.diagonal() * 2)  # diagonal más fuerte
        channel_matrices.append(H)
    
    return channel_matrices


def train_with_sb3(
    n_APs=5,
    num_channels=3,
    P0=4,
    n_power_levels=2,
    Pmax=0.7,
    total_timesteps=500000,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    building_id=990,
    b5g=False
):
    """
    Entrena el modelo usando PPO de Stable-Baselines3
    """
    
    # Cargar matrices de canal
    print("Cargando matrices de canal...")
    train_matrices = load_channel_matrices(building_id, b5g, train=True)
    val_matrices = load_channel_matrices(building_id, b5g, train=False)
    
    # Crear iteradores
    train_iterator = make_channel_iterator(train_matrices)
    val_iterator = make_channel_iterator(val_matrices)
    
    # Crear entornos
    print("Creando entornos...")
    train_env = APNetworkEnv(
        n_APs=n_APs,
        num_channels=num_channels,
        P0=P0,
        n_power_levels=n_power_levels,
        Pmax=Pmax,
        max_steps=500,
        H_iterator=train_iterator
    )
    
    # Wrapper para monitorear
    train_env = Monitor(train_env)
    
    # Verificar que el entorno es válido
    print("Verificando entorno...")
    check_env(train_env, warn=True)
    
    # Crear entorno de validación
    val_env = APNetworkEnv(
        n_APs=n_APs,
        num_channels=num_channels,
        P0=P0,
        n_power_levels=n_power_levels,
        Pmax=Pmax,
        max_steps=500,
        H_iterator=val_iterator
    )
    val_env = Monitor(val_env)
    
    # Configurar callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/eval',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./logs/checkpoints',
        name_prefix='ppo_apnetwork'
    )
    
    # Configurar política personalizada con GNN
    policy_kwargs = dict(
        n_APs=n_APs,
        gnn_hidden_dim=32,
        gnn_num_layers=3,
        K=3
    )
    
    # Crear modelo PPO con tu policy personalizada
    print("Creando modelo PPO...")
    model = PPO(
        GNNActorCriticPolicy,
        train_env,
        learning_rate=learning_rate,
        n_steps=2048,  # pasos por actualización
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # exploración
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )
    
    # Entrenar
    print("Iniciando entrenamiento...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Guardar modelo final
    model.save("./logs/ppo_apnetwork_final")
    print("Modelo guardado en ./logs/ppo_apnetwork_final")
    
    return model


def test_model(model_path, n_episodes=10):
    """
    Testea un modelo entrenado
    """
    # Cargar modelo
    model = PPO.load(model_path)
    
    # Crear entorno de test
    test_matrices = load_channel_matrices(train=False)
    test_iterator = make_channel_iterator(test_matrices)
    
    test_env = APNetworkEnv(
        n_APs=5,
        num_channels=3,
        P0=4,
        n_power_levels=2,
        Pmax=0.7,
        max_steps=500,
        H_iterator=test_iterator
    )
    
    # Evaluar
    episode_rewards = []
    avg_powers_list = []
    
    for ep in range(n_episodes):
        obs, info = test_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        if 'avg_powers' in info:
            avg_powers_list.append(info['avg_powers'])
        
        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}")
    
    print(f"\nPromedio de recompensas: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Promedio de potencias: {np.mean(avg_powers_list, axis=0)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento con SB3')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_path', type=str, default='./logs/ppo_apnetwork_final')
    parser.add_argument('--n_APs', type=int, default=5)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_with_sb3(
            n_APs=args.n_APs,
            num_channels=args.num_channels,
            total_timesteps=args.timesteps,
            learning_rate=args.lr
        )
    else:
        test_model(args.model_path)
