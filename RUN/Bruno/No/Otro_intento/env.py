import gymnasium as gym
from gymnasium import spaces
import numpy as np
from NetROML.RUN.Bruno.No.rates import get_reward

class APNetworkEnv(gym.Env):
    """
    Entorno de red inalámbrica para la asignación de canales y potencias en APs.
    Observation space real será el grafo (wrapper) para la GNN.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_APs=5, num_channels=3, P0=4, n_power_levels=3, power_levels_explicit=None, Pmax=0.7, max_steps=500):
        super().__init__()
        self.n_APs = n_APs
        self.num_channels = num_channels
        self.max_steps = max_steps

        # Power levels
        if power_levels_explicit is not None:
            self.power_levels = power_levels_explicit
            self.n_power_levels = len(self.power_levels)
            self.P0 = max(self.power_levels)
        else:
            self.n_power_levels = n_power_levels
            self.P0 = P0
            self.power_levels = (np.arange(1, self.n_power_levels + 1)/self.n_power_levels)*self.P0

        self.Pmax = P0 * Pmax

        # Action space: MultiDiscrete
        self.action_space = spaces.MultiDiscrete(
            [(self.num_channels * self.n_power_levels + 1)] * self.n_APs
        )

        # Observation space indicativo
        self.observation_space = "torch_geometric.Data"  # REAL: usamos grafos como observación

        self.state = None
        self.current_step = 0
        self.power_history = []



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Estado interno inicial (solo para reward)
        canales = np.random.randint(1, self.num_channels + 1, size=(self.n_APs, 1)).astype(np.float32)
        indices_pot = np.random.randint(0, self.n_power_levels, size=(self.n_APs, 1))
        potencias = self.power_levels[indices_pot].astype(np.float32)
        self.state = np.hstack([canales, potencias])

        self.power_history = []
        return self.state, {}  # el wrapper se encarga de transformar a Data



    def step(self, action):
        self.current_step += 1
        action = np.array(action)

        new_state = np.zeros((self.n_APs, 2), dtype=np.float32)
        active_mask = action > 0
        a_adj = action[active_mask] - 1

        canales = a_adj // self.n_power_levels + 1

        indices_asignacion_potencias = a_adj % self.n_power_levels
        potencias = self.power_levels[indices_asignacion_potencias].astype(np.float32)
        
        new_state[active_mask, 0] = canales
        new_state[active_mask, 1] = potencias

        powers_this_step = np.zeros(self.n_APs, dtype=np.float32)
        powers_this_step[active_mask] = potencias
        self.power_history.append(powers_this_step)

        self.state = new_state

        # Reward (tu función)
        reward = get_reward(self.power_history, self.Pmax, phi=None, channel_matrix_batch=None, sigma=None, p0=self.P0, alpha=0.3)

        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {"avg_power": np.mean(self.power_history, axis=0)}

        return self.state, reward, terminated, truncated, info



    def render(self):
        print(f"Step {self.current_step} | State:\n{self.state}")



    def close(self):
        pass
