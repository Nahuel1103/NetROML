import gymnasium as gym
from gymnasium import spaces
import numpy as np

class APNetworkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_APs=5, num_channels=3, max_power=2, max_steps=50, power_levels=3):
        super().__init__()

        self.n_APs = n_APs
        self.num_channels = num_channels
        self.max_power = max_power
        self.max_steps = max_steps
        self.power_levels=power_levels

        # Cada AP: {0=apagado, 1, ..., num_channels} con niveles de potencia
        # action_space recibe una lista con una entrada por cada ap (*self.n_APs), 
        # y cada entrada tiene la cantidad maxima de opciones que puede tomar (self.num_channels * self.power_levels + apagado).
        self.action_space = spaces.MultiDiscrete([(self.num_channels * self.power_levels + 1)] * self.n_APs)
        
        # Estado de la red: [canal (1..num_channels), potencia (1..power_levels)] por AP
        # Los valres del obs space son float por la GNN, pero el step deberia redondear.
        self.observation_space = spaces.Box(
            low=np.array([1, 1], dtype=np.float32),
            high=np.array([self.num_channels, self.power_levels], dtype=np.float32)
        )

        # ESTO ESTA COMENTADO POR SI LA INTERFERENCIA ES PARTE DEL ESTADO

        # Estado de la red: [canal (1..num_channels), potencia (1..power_levels), interferencia ¿¿(0..1)??] por AP
        # Los valres del obs space son float por la GNN, pero el step deberia redondear.
        # self.observation_space = spaces.Box(
        #     low=np.array([1, 1, 0], dtype=np.float32),
        #     high=np.array([self.num_channels, self.power_levels, 1], dtype=np.float32)
        # )


        self.state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Estado inicial aleatorio 
        
        # Canal: 1..num_channels
        canales = np.random.randint(1, self.num_channels + 1, size=(self.n_APs, 1), dtype=np.float32)
        # Potencia: 1..power_levels
        potencias = np.random.randint(1, self.power_levels + 1, size=(self.n_APs, 1), dtype=np.float32)
        
        # ESTO ESTA COMENTADO POR SI LA INTERFERENCIA ES PARTE DEL ESTADO
        # Interferencia: 0..1
        # interferencia = np.random.rand(self.n_APs, 1).astype(np.float32)
        # self.state = np.hstack([canales, potencias, interferencia])
        
        self.state = np.hstack([canales, potencias])

        return self.state, {}

    


    def step(self, action):
        self.current_step += 1

        # Simular efectos de la acción en el estado
        # (aquí tendrías que definir cómo la acción cambia la red)
        self.state = np.random.rand(self.n_APs, 4).astype(np.float32)

        # Calcular reward (ejemplo simple)
        throughput = np.sum(self.state[:, 3])  # suponer col 3 = tráfico
        interference = np.sum(self.state[:, 2])  # suponer col 2 = interferencia
        reward = throughput - 0.5 * interference

        # Condiciones de fin
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Step {self.current_step} | State:\n{self.state}")

    def close(self):
        pass
