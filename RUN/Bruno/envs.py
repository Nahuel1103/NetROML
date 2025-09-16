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
        self.action_space = spaces.MultiDiscrete(
            [(self.num_channels * self.power_levels + 1)] * self.n_APs
        )
        
        # Estado de la red: [canal (1..num_channels), potencia (1..power_levels)] por AP
        # Los valres del obs space son float por la GNN, pero el step deberia redondear.
        self.observation_space = spaces.Box(
            low=np.array([[0, 0]] * self.n_APs, dtype=np.float32),
            high=np.array([[self.num_channels, self.power_levels]] * self.n_APs, dtype=np.float32),
            dtype=np.float32
        )

        # ESTO ESTA COMENTADO POR SI LA INTERFERENCIA ES PARTE DEL ESTADO

        # Estado de la red: [canal (1..num_channels), potencia (1..power_levels), interferencia ¿¿(0..1)??] por AP
        # Los valres del obs space son float por la GNN, pero el step deberia redondear.
        # self.observation_space = spaces.Box(
        #     low=np.array([[0, 0, 0]] * self.n_APs, dtype=np.float32),
        #     high=np.array([[self.num_channels, self.power_levels,1]] * self.n_APs, dtype=np.float32),
        #     dtype=np.float32
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


    
    def step(self, action, H=None, graph=None):
        self.current_step += 1
        action = np.array(action)

        # actualizar H y graph si se pasa
        if H is not None:
            self.H = H
        if graph is not None:
            self.graph = graph

        # Inicializamos estado nuevo con ceros
        new_state = np.zeros((self.n_APs, 2), dtype=np.float32)

        # Máscara: cuáles APs están activos (no apagado)
        active_mask = action > 0

        # Ajustamos acciones (para que 0 quede apagado y el resto arranque desde 0)
        a_adj = action[active_mask] - 1

        # Calculamos canal y potencia solo para los activos
        canales = a_adj // self.power_levels + 1
        potencias = a_adj % self.power_levels + 1

        # Asignamos de una sola vez
        new_state[active_mask, 0] = canales
        new_state[active_mask, 1] = potencias

        # Este será el siguiente estado de la red
        self.state = new_state

        # Reward de ejemplo: suma de potencias activas
        # HAY QUE EDITARLO
        reward = np.sum(self.state[:, 1])

        # terminated se usaría si, cumplida una condición, debe empezar el siguiente paso reseteando el state.
        terminated = False

        # Indica si el episodio terminó porque se alcanzó un limite impuesto (por ejemplo, max steps)
        truncated = self.current_step >= self.max_steps

        # {} es un diccionario opcional que se usa para devolver información extra que no forma parte del estado, 
        # pero puede ser útil para debugging, logging o métricas.
        return self.state, reward, terminated, truncated, {}



    def render(self):
        # No es obligatorio, pero útil si se quiere visualizar lo que pasa (debugging o mostrar resultados).
        # Puede ser tan simple como un print o tan complejo como gráficos en tiempo real.
        # Si no lo necesitás, podés dejarlo vacío o no implementarlo.
        print(f"Step {self.current_step} | State:\n{self.state}")



    def close(self):
        # Tampoco es obligatorio.
        # Se usa si tu entorno abre recursos externos (ventanas gráficas, archivos, conexiones, etc.).
        # Si no abrís nada, podés dejarlo como pass.
        pass
