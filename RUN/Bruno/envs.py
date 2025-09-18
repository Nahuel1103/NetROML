import gymnasium as gym
from gymnasium import spaces
import numpy as np

class APNetworkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_APs=5, num_channels=3, max_power=2, max_steps=50, power_levels=3, flatten_obs=False):
        super().__init__()

        self.n_APs = n_APs
        self.num_channels = num_channels
        self.max_power = max_power
        self.max_steps = max_steps
        self.power_levels=power_levels
        self.flatten_obs = flatten_obs



        # DEFINO ESPACIO DE ACCIONES

        # Cada AP: {0=apagado, 1, ..., num_channels} con niveles de potencia
        # action_space recibe una lista con una entrada por cada ap (*self.n_APs), 
        # y cada entrada tiene la cantidad maxima de opciones que puede tomar (self.num_channels * self.power_levels + apagado).
        self.action_space = spaces.MultiDiscrete(
            [(self.num_channels * self.power_levels + 1)] * self.n_APs
        )
        

        # DEFINO ESPACIO DE OBSERVACIÓN
        if self.flatten_obs:
            # Observaciones como vector 1D
            self.observation_space = spaces.Box(
                low=np.zeros(self.n_APs * 2, dtype=np.float32),
                high=np.array([self.num_channels, self.power_levels] * self.n_APs, dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Observaciones como matriz (n_APs, 2)
            self.observation_space = spaces.Box(
                low=np.array([[0, 0]] * self.n_APs, dtype=np.float32),
                high=np.array([[self.num_channels, self.power_levels]] * self.n_APs, dtype=np.float32),
                dtype=np.float32
            )


        self.state = None
        self.current_step = 0



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Estado inicial aleatorio
         
        # Canal: 1..num_channels
        canales = np.random.randint(1, self.num_channels + 1, size=(self.n_APs, 1)).astype(np.float32)
        # Potencia: 1..power_levels
        potencias = np.random.randint(1, self.power_levels + 1, size=(self.n_APs, 1)).astype(np.float32)
        

        
        state = np.hstack([canales, potencias])

        if self.flatten_obs:
            state = state.flatten()

        self.state = state

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
        if self.flatten_obs:
            self.state = new_state.flatten()
        else:
            self.state = new_state


        # Reward de ejemplo: suma de potencias activas
        # HAY QUE EDITARLO
        # reward = float(np.sum(self.state[:, 1]))
        reward = float(np.sum(new_state[:, 1]))

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
