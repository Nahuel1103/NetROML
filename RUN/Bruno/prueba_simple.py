from stable_baselines3 import PPO
from envs import APNetworkEnv

# Creo el entorno (el nuestro custom)
env = APNetworkEnv(n_APs=5, num_channels=3, power_levels=2, flatten_obs=True)

# Creo el agente
agent = PPO("MlpPolicy", env, verbose=1)

# Entrenar un poquito (timesteps bajos solo para test)
agent.learn(total_timesteps=5000)

# Probar agente entrenado
obs, _ = env.reset()
print("Estado inicial:", obs)

for step in range(5):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"\nStep {step+1}")
    print("Acción elegida:", action)
    print("Nuevo estado:", obs)
    print("Reward:", reward)
    if terminated or truncated:
        break
