# Mini script que chequea si un entorno cumple con todo lo requerido por la API de Gymnasium
import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.env_checker import check_env

# Importá tu clase desde envs.py
# from NetROML.RUN.Bruno.No.envs1 import APNetworkEnv
from envs import APNetworkEnv

# Crear instancia de tu entorno 
# (tener en cuenta los parametros necesarios para la implementación)
# Generar 5 matrices 4x4 con enteros aleatorios entre 1 y 9
n_APs = 4
H_random_list = [np.random.randint(1, 10, size=(n_APs, n_APs)) for _ in range(5)]
H_iter = iter(H_random_list)  # iterador compartido

env = APNetworkEnv(n_APs=n_APs, H_iterator=H_iter)

# Verificar que cumple la API de Gymnasium
check_env(env, warn=True)

print("-------------------------------------------------------------")
print("¡El entorno está correcto! Suerte en pila con lo que sigue...")
print("-------------------------------------------------------------")
