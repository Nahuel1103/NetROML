# Mini script que chequea si un entorno cumple con todo lo requerido por la API de Gymnasium
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.env_checker import check_env

# Importá tu clase desde envs.py
# from NetROML.RUN.Bruno.No.envs1 import APNetworkEnv
from envs import APNetworkEnv

# Crear instancia de tu entorno
env = APNetworkEnv()

# Verificar que cumple la API de Gymnasium
check_env(env, warn=True)

print("-------------------------------------------------------------")
print("¡El entorno está correcto! Suerte en pila con lo que sigue...")
print("-------------------------------------------------------------")
