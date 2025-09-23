from wifi_env import *

# Crear entorno
config = WirelessConfig(num_links=5, num_channels=5)
env = WirelessGNNEnvironment(config)

# Entrenar con cualquier algoritmo de RL
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)