"""
Script de entrenamiento usando Stable-Baselines3 con tu environment
SIN BATCHES - el environment procesa un grafo a la vez
"""
import numpy as np
import torch
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import APNetworkEnv
from custom_policy import GNNActorCriticPolicy


class PowerConstraintCallback(BaseCallback):
    """
    Callback personalizado para monitorear el constraint de potencia
    y los multiplicadores de Lagrange (mu)
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.power_violations = []
        self.mu_values = []
        self.rewards = []
        
    def _on_step(self) -> bool:
        # Acceder al environment
        if hasattr(self.training_env, 'get_attr'):
            # VecEnv
            mu = self.training_env.get_attr('mu_power')[0]
            if hasattr(self.training_env.get_attr('power_history')[0], '__len__'):
                if len(self.training_env.get_attr('power_history')[0]) > 0:
                    avg_power = np.mean(self.training_env.get_attr('power_history')[0], axis=0)
                    self.power_violations.append(avg_power)
        else:
            # Env normal
            mu = self.training_env.mu_power
            if len(self.training_env.power_history) > 0:
                avg_power = np.mean(self.training_env.power_history, axis=0)
                self.power_violations.append(avg_power)
        
        self.mu_values.append(mu.copy() if isinstance(mu, np.ndarray) else mu.detach().cpu().numpy())
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Llamado al final de cada rollout"""
        if self.verbose > 0 and len(self.mu_values) > 0:
            avg_mu = np.mean(self.mu_values[-100:], axis=0)
            print(f"Avg mu (últimos 100 steps): {avg_mu}")


def make_channel_iterator(channel_matrices):
    """
    Crea un iterador infinito de matrices de canal
    Importante: cuando se acaben, vuelve a empezar
    """
    while True:
        for H in channel_matrices:
            yield H


def load_channel_matrices_real(building_id=990, b5g=False, train=True, num_links=5):
    """
    Carga las matrices de canal REALES desde tus datos
    Integra con tu código existente
    """
    try:
        from utils import graphs_to_tensor, graphs_to_tensor_synthetic, get_gnn_inputs
        import torch
        
        # Intenta cargar datos sintéticos primero
        try:
            x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
                num_links=num_links,
                num_features=1,
                b5g=b5g,
                building_id=building_id
            )
            if train:
                channel_matrices = channel_matrix_tensor[:7000].numpy()
            else:
                channel_matrices = channel_matrix_tensor[7000:].numpy()
            
            print(f"Cargadas {len(channel_matrices)} matrices de canal sintéticas")
            return [H.astype(np.float32) for H in channel_matrices]
            
        except Exception as e:
            print(f"No se pudieron cargar datos sintéticos: {e}")
            
        # Si falla, intenta datos reales
        x_tensor, channel_matrix_tensor = graphs_to_tensor(
            train=train,
            num_links=num_links,
            num_features=1,
            b5g=b5g,
            building_id=building_id
        )
        
        channel_matrices = channel_matrix_tensor.numpy()
        print(f"Cargadas {len(channel_matrices)} matrices de canal reales")
        return [H.astype(np.float32) for H in channel_matrices]
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        print("Usando datos sintéticos aleatorios")
        return load_channel_matrices_dummy(num_links, train)


def load_channel_matrices_dummy(num_links=5, train=True):
    """
    Genera matrices de canal sintéticas para testing
    """
    n_samples = 7000 if train else 1000
    
    channel_matrices = []
    for _ in range(n_samples):
        H = np.random.rand(num_links, num_links).astype(np.float32)
        # Diagonal más fuerte (señal directa)
        np.fill_diagonal(H, H.diagonal() * 3)
        # Normalizar
        H = H / np.max(H)
        channel_matrices.append(H)
    
    return channel_matrices


def train_with_sb3(
    n_APs=5,
    num_channels=3,
    P0=4,
    n_power_levels=2,
    power_levels_explicit=None,
    Pmax=0.7,
    total_timesteps=500000,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    building_id=990,
    b5g=False,
    algorithm='PPO',
    use_real_data=False,
    save_dir='./logs'
):
    """
    Entrena el modelo usando PPO o A2C de Stable-Baselines3
    
    Args:
        algorithm: 'PPO' o 'A2C'
        use_real_data: si True, intenta cargar tus datos reales
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Cargar matrices de canal
    print("Cargando matrices de canal...")
    if use_real_data:
        train_matrices = load_channel_matrices_real(building_id, b5g, train=True, num_links=n_APs)
        val_matrices = load_channel_matrices_real(building_id, b5g, train=False, num_links=n_APs)
    else:
        train_matrices = load_channel_matrices_dummy(num_links=n_APs, train=True)
        val_matrices = load_channel_matrices_dummy(num_links=n_APs, train=False)
    
    # Crear iteradores (infinitos para que no se agoten)
    train_iterator = make_channel_iterator(train_matrices)
    val_iterator = make_channel_iterator(val_matrices)
    
    # Crear entornos
    print("Creando entorno de entrenamiento...")
    train_env = APNetworkEnv(
        n_APs=n_APs,
        num_channels=num_channels,
        P0=P0,
        n_power_levels=n_power_levels,
        power_levels_explicit=power_levels_explicit,
        Pmax=Pmax,
        max_steps=100,  # episodios más cortos para RL
        H_iterator=train_iterator
    )
    
    # Wrapper para monitorear
    train_env = Monitor(train_env, filename=f"{save_dir}/train_monitor.csv")
    
    # Verificar que el entorno es válido
    print("Verificando entorno...")
    try:
        check_env(train_env, warn=True)
        print("✓ Entorno válido")
    except Exception as e:
        print(f"⚠ Advertencia en verificación del entorno: {e}")
    
    # Crear entorno de validación
    val_env = APNetworkEnv(
        n_APs=n_APs,
        num_channels=num_channels,
        P0=P0,
        n_power_levels=n_power_levels,
        power_levels_explicit=power_levels_explicit,
        Pmax=Pmax,
        max_steps=100,
        H_iterator=val_iterator
    )
    val_env = Monitor(val_env, filename=f"{save_dir}/val_monitor.csv")
    
    # Configurar callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=f'{save_dir}/best_model',
        log_path=f'{save_dir}/eval',
        eval_freq=5000,  # evaluar cada 5000 steps
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f'{save_dir}/checkpoints',
        name_prefix=f'{algorithm.lower()}_apnetwork'
    )
    
    power_callback = PowerConstraintCallback(verbose=1)
    
    # Configurar política personalizada con GNN
    policy_kwargs = dict(
        n_APs=n_APs,
        gnn_hidden_dim=32,
        gnn_num_layers=3,
        K=3,
        net_arch=dict(
            pi=[64, 64],  # actor
            vf=[64, 64]   # critic
        )
    )
    
    # Crear modelo
    print(f"Creando modelo {algorithm}...")
    if algorithm == 'PPO':
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
            tensorboard_log=f"{save_dir}/tensorboard/"
        )
    elif algorithm == 'A2C':
        model = A2C(
            GNNActorCriticPolicy,
            train_env,
            learning_rate=learning_rate,
            n_steps=5,  # A2C usa menos steps
            gamma=gamma,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"{save_dir}/tensorboard/"
        )
    else:
        raise ValueError(f"Algoritmo {algorithm} no soportado")
    
    # Entrenar
    print("Iniciando entrenamiento...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, power_callback],
        progress_bar=True
    )
    
    # Guardar modelo final
    final_path = f"{save_dir}/{algorithm.lower()}_apnetwork_final"
    model.save(final_path)
    print(f"✓ Modelo guardado en {final_path}")
    
    return model, power_callback


def test_model(model_path, n_episodes=10, n_APs=5, num_channels=3, 
               P0=4, n_power_levels=2, Pmax=0.7, use_real_data=False):
    """
    Testea un modelo entrenado
    """
    print(f"Cargando modelo desde {model_path}...")
    
    # Detectar algoritmo del path
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    elif 'a2c' in model_path.lower():
        model = A2C.load(model_path)
    else:
        print("No se pudo detectar algoritmo, intentando PPO...")
        model = PPO.load(model_path)
    
    # Cargar matrices de test
    if use_real_data:
        test_matrices = load_channel_matrices_real(train=False, num_links=n_APs)
    else:
        test_matrices = load_channel_matrices_dummy(num_links=n_APs, train=False)
    
    test_iterator = make_channel_iterator(test_matrices)
    
    test_env = APNetworkEnv(
        n_APs=n_APs,
        num_channels=num_channels,
        P0=P0,
        n_power_levels=n_power_levels,
        Pmax=Pmax,
        max_steps=100,
        H_iterator=test_iterator
    )
    
    # Evaluar
    episode_rewards = []
    avg_powers_list = []
    mu_final_list = []
    
    print(f"\nEvaluando {n_episodes} episodios...")
    for ep in range(n_episodes):
        obs, info = test_env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        if 'avg_powers' in info:
            avg_powers_list.append(info['avg_powers'])
        
        mu_final = test_env.mu_power
        mu_final_list.append(mu_final.copy() if isinstance(mu_final, np.ndarray) else mu_final)
        
        print(f"  Episode {ep+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    print("\n" + "="*50)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*50)
    print(f"Recompensa promedio: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Recompensa máxima: {np.max(episode_rewards):.2f}")
    print(f"Recompensa mínima: {np.min(episode_rewards):.2f}")
    
    if len(avg_powers_list) > 0:
        avg_power_mean = np.mean(avg_powers_list, axis=0)
        print(f"\nPotencia promedio por AP: {avg_power_mean}")
        print(f"Constraint Pmax: {Pmax}")
        violations = np.sum(avg_power_mean > Pmax)
        print(f"APs violando constraint: {violations}/{n_APs}")
    
    if len(mu_final_list) > 0:
        mu_mean = np.mean(mu_final_list, axis=0)
        print(f"\nMultiplicadores mu finales (promedio): {mu_mean}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento con SB3')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_path', type=str, default='./logs/ppo_apnetwork_final')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'A2C'])
    parser.add_argument('--n_APs', type=int, default=5)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--n_power_levels', type=int, default=2)
    parser.add_argument('--P0', type=float, default=4.0)
    parser.add_argument('--Pmax', type=float, default=0.7)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_real_data', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./logs')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n" + "="*50)
        print("CONFIGURACIÓN DE ENTRENAMIENTO")
        print("="*50)
        print(f"Algoritmo: {args.algorithm}")
        print(f"APs: {args.n_APs}")
        print(f"Canales: {args.num_channels}")
        print(f"Niveles de potencia: {args.n_power_levels}")
        print(f"P0: {args.P0}")
        print(f"Pmax: {args.Pmax * args.P0}")
        print(f"Total timesteps: {args.timesteps}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size: {args.batch_size}")
        print(f"Datos reales: {args.use_real_data}")
        print("="*50 + "\n")
        
        model, callback = train_with_sb3(
            n_APs=args.n_APs,
            num_channels=args.num_channels,
            P0=args.P0,
            n_power_levels=args.n_power_levels,
            Pmax=args.Pmax,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            algorithm=args.algorithm,
            use_real_data=args.use_real_data,
            save_dir=args.save_dir
        )
        
        print("\n✓ Entrenamiento completado")
        
        # Test rápido del modelo entrenado
        print("\nEvaluación rápida del modelo entrenado...")
        test_model(
            model_path=f"{args.save_dir}/{args.algorithm.lower()}_apnetwork_final",
            n_episodes=5,
            n_APs=args.n_APs,
            num_channels=args.num_channels,
            P0=args.P0,
            n_power_levels=args.n_power_levels,
            Pmax=args.Pmax * args.P0,
            use_real_data=args.use_real_data
        )
        
    else:
        test_model(
            model_path=args.model_path,
            n_episodes=20,
            n_APs=args.n_APs,
            num_channels=args.num_channels,
            P0=args.P0,
            n_power_levels=args.n_power_levels,
            Pmax=args.Pmax * args.P0,
            use_real_data=args.use_real_data
        )
