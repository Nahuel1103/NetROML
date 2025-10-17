from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation

env = FlattenObservation(APNetworkEnv())

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e5)
