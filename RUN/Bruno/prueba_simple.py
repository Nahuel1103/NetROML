# Algoritmo muy simple para comprobar como funciona el entorno creado


from stable_baselines3 import PPO
from envs import APNetworkEnv
import numpy as np


n_APs = 4
H_random_list = [np.random.randint(0, 5, size=(n_APs, n_APs)) for _ in range(10000)]
H_iter = iter(H_random_list)  # iterador compartido


# Creo el entorno (el nuestro custom)
env = APNetworkEnv(n_APs=n_APs, H_iterator=H_iter)


# Creo el agente
agent = PPO("MultiInputPolicy", env, verbose=1)

# Entrenar un poquito (timesteps bajos solo para test)
agent.learn(total_timesteps=1)

# Probar agente entrenado
obs, _ = env.reset()
print("Estado inicial:", obs)

for step in range(10000):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"\nStep {step+1}")
    print("Acción elegida:", action)
    print("Nuevo estado:", obs)
    print("Reward:", reward)
    if terminated or truncated:
        break
