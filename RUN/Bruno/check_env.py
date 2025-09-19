# Mini script que chequea si un entorno cumple con todo lo requerido por la API de Gymnasium


from stable_baselines3.common.env_checker import check_env

# Importá tu clase desde envs.py
from envs import APNetworkEnv

# Crear instancia de tu entorno
env = APNetworkEnv(flatten_obs=True)

# Verificar que cumple la API de Gymnasium
check_env(env, warn=True)

print("-------------------------------------------------------------")
print("¡El entorno está correcto! Suerte en pila con lo que sigue...")
print("-------------------------------------------------------------")
