from wifi_env import WifiEnv
import numpy as np

# Crear entorno
env = WifiEnv()

obs, info = env.reset()
print("Estado inicial:", obs)

obs, info = env.reset()
print("Estado inicial:", obs)

for _ in range(5):
    action = env.action_space.sample()  # vector plano [num_links*3]
    obs, reward, done, truncated, info = env.step(action)
    print(f"Acción: {action}, Recompensa: {reward}, Nuevo estado: {obs}")

