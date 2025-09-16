import gymnasium as gym
import numpy as np
from gymnasium import spaces

class WifiChannelEnv(gym.Env):
    """
    Entorno Gymnasium para selección de canal y potencia en redes WiFi con GNN.
    Cada step representa un time slot.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, num_links=5, num_channels=3, num_power_levels=2, max_antenna_power=4.0, channel_matrix=None, initial_state=None):
        super().__init__()
        self.num_links = num_links
        self.num_channels = num_channels
        self.num_power_levels = num_power_levels
        self.max_antenna_power = max_antenna_power

        # Acción: para cada AP, elegir (canal, potencia)
        # Acción discreta: (num_channels * num_power_levels + 1) por AP (incluye no transmitir)
        self.action_space = spaces.MultiDiscrete([
            1 + num_channels * num_power_levels for _ in range(num_links)
        ])

        # Observación: estado de APs + matriz de canales
        # Estado de APs: vector binario (on/off) o potencia actual
        # Matriz de canales: (num_links, num_links)
        self.observation_space = spaces.Dict({
            "ap_state": spaces.Box(low=0, high=max_antenna_power, shape=(num_links,), dtype=np.float32),  #No se contempla el canal
            "channel_matrix": spaces.Box(low=0, high=np.inf, shape=(num_links, num_links), dtype=np.float32)
        })

        # Dataset de validación opcional
        self.channel_matrix = channel_matrix
        self.state = initial_state
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Inicializar estado de APs y matriz de canales
        # TODO: cargar o muestrear channel_matrix y estado inicial
        self.current_step = 0
        if self.channel_matrix is None:
            self.channel_matrix = np.ones((self.num_links, self.num_links), dtype=np.float32)
        self.state = np.zeros((self.num_links,), dtype=np.float32)
        obs = {"ap_state": self.state.copy(), "channel_matrix": self.channel_matrix.copy()}
        return obs, {}

    def step(self, action):
        # TODO: aplicar acción, actualizar estado, calcular reward y penalización
        # action: array de shape (num_links,)
        # Decodificar acción: para cada AP, (no_tx) o (canal, potencia)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        # TODO: lógica de transición y reward
        obs = {"ap_state": self.state.copy(), "channel_matrix": self.channel_matrix.copy()}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Opcional: mostrar estado actual
        print(f"Step: {self.current_step}, AP state: {self.state}")

    def close(self):
        pass
