from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Importá tu clase desde envs.py
from envs import APNetworkEnv

# Crear instancia de tu entorno
env = APNetworkEnv()

# Verificar que cumple la API de Gymnasium
check_env(env, warn=True)