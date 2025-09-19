import gymnasium as gym
from gymnasium import spaces
import numpy as np

class APNetworkEnv(gym.Env):
    """
    Entorno de red inalámbrica para la asignación de canales y potencias en puntos de acceso (APs).
    
    Este entorno simula varios APs que deben seleccionar canal y nivel de potencia en cada paso,
    bajo la restricción de una potencia máxima relativa (Pmax) a potencia máxima a la que pueden
    transmitir los APs (P0). 
    
    Compatible con Gymnasium.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_APs=5, num_channels=3, P0=2, n_power_levels=3, power_levels_explicit=None, Pmax=0.7, max_steps=50, flatten_obs=False):
        """
        Inicializa el entorno de red de APs.

        Parámetros
        ----------
        n_APs : int
            Número de puntos de acceso (APs) en la red.
        num_channels : int
            Número de canales disponibles para los APs.
        P0 : float
            Potencia máxima a la que pueden transmitir los APs.
        n_power_levels : int
            Número de niveles discretos de potencia que cada AP puede usar.
        power_levels_explicit : array-like, opcional
            Lista de niveles de potencia explícitos a usar en lugar de uniformes.
            Si se pasa, se ignoran P0 y n_power_levels.
        Pmax : float
            Fracción de P0 que representa la potencia máxima permitida (ej. 0.7 = 70% de P0).
        max_steps : int
            Número máximo de pasos en un episodio.
        flatten_obs : bool
            Si True, devuelve las observaciones como vector 1D; si False, como matriz (n_APs, 2).
            Útil para modelos como PPO.
        """
        super().__init__()

        self.n_APs = n_APs
        self.num_channels = num_channels
        self.Pmax = Pmax
        self.max_steps = max_steps
        self.flatten_obs = flatten_obs

        if power_levels_explicit is not None:
            self.power_levels=power_levels_explicit
            self.n_power_levels=len(self.power_levels)
            self.P0 = max(self.power_levels)

        else:
            self.n_power_levels=n_power_levels
            self.P0 = P0
            self.power_levels = (np.arange(1, self.n_power_levels + 1) / self.n_power_levels) * self.P0



        # DEFINO ESPACIO DE ACCIONES

        # Cada AP: {0=apagado, 1, ..., num_channels} con niveles de potencia
        # action_space recibe una lista con una entrada por cada ap (*self.n_APs), 
        # y cada entrada tiene la cantidad maxima de opciones que puede tomar (self.num_channels * self.n_power_levels + apagado).
        self.action_space = spaces.MultiDiscrete(
            [(self.num_channels * self.n_power_levels + 1)] * self.n_APs
        )
        

        # DEFINO ESPACIO DE OBSERVACIÓN
        if self.flatten_obs:
            # Observaciones como vector 1D
            self.observation_space = spaces.Box(
                low=np.zeros(self.n_APs * 2, dtype=np.float32),
                high=np.array([self.num_channels, self.P0] * self.n_APs, dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Observaciones como matriz (n_APs, 2)
            self.observation_space = spaces.Box(
                low=np.array([[0, 0]] * self.n_APs, dtype=np.float32),
                high=np.array([[self.num_channels, self.P0]] * self.n_APs, dtype=np.float32),
                dtype=np.float32
            )


        self.state = None
        self.current_step = 0



    def reset(self, seed=None, options=None):
        """
        Resetea el entorno a un estado inicial aleatorio.

        Parámetros
        ----------
        seed : int, opcional
            Semilla para reproducibilidad.
        options : dict, opcional
            Opciones adicionales (no usadas actualmente).

        Returns
        -------
        state : np.ndarray
            Estado inicial de la red, con canales y potencias.
        info : dict
            Diccionario vacío para compatibilidad con Gymnasium.
        """

        super().reset(seed=seed)
        self.current_step = 0

        # Estado inicial aleatorio
         
        # Canal: 1..num_channels
        canales = np.random.randint(1, self.num_channels + 1, size=(self.n_APs, 1)).astype(np.float32)
        # Potencia: 1..n_power_levels
        indices_pot = np.random.randint(0, self.n_power_levels, size=(self.n_APs, 1))
        potencias = self.power_levels[indices_pot]
        
        state = np.hstack([canales, potencias])

        if self.flatten_obs:
            state = state.flatten()

        self.state = state

        return self.state, {}


    
    def step(self, action, H=None, graph=None):
        """
        Avanza el entorno un paso, dado un vector de acciones.

        Parámetros
        ----------
        action : array-like
            Vector de acciones de tamaño n_APs. Cada elemento indica el canal y nivel de potencia elegido 
            (0 = apagado, 1..num_channels*power_levels = distintos niveles y canales).
        H : np.ndarray, opcional
            Matriz de atenuaciones entre APs (opcional).
        graph : np.ndarray, opcional
            Representación de la topología de la red (opcional).

        Returns
        -------
        state : np.ndarray
            Nuevo estado del entorno tras aplicar la acción.
        reward : float
            Recompensa obtenida en este paso.
        terminated : bool
            True si el episodio terminó por alguna condición de finalización.
        truncated : bool
            True si el episodio terminó por alcanzar `max_steps`.
        info : dict
            Diccionario opcional con información adicional (vacío actualmente).
        """

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

        # Hallo que canal usan (solo para los activos)
        canales = a_adj // self.n_power_levels + 1

        # Vemos que potencia usan (solo para los activos)
        indices_asignacion_potencias = a_adj % self.n_power_levels        # potencias que eligió cada uno (indices en el array de posibles)
        potencias = self.power_levels[indices_asignacion_potencias]        # potencas reales pero que eliguó cada uno

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
        reward = float(np.sum(new_state[:, 1]))
        

        # terminated se usaría si, cumplida una condición, debe empezar el siguiente paso reseteando el state.
        terminated = False

        # Indica si el episodio terminó porque se alcanzó un limite impuesto (por ejemplo, max steps)
        truncated = self.current_step >= self.max_steps

        # {} es un diccionario opcional que se usa para devolver información extra que no forma parte del estado, 
        # pero puede ser útil para debugging, logging o métricas.
        return self.state, reward, terminated, truncated, {}



    def render(self):
        """
        Muestra el estado actual del entorno.

        Para este entorno, simplemente imprime el estado con canales y potencias de cada AP.
        """
        # No es obligatorio, pero útil si se quiere visualizar lo que pasa (debugging o mostrar resultados).
        # Puede ser tan simple como un print o tan complejo como gráficos en tiempo real.
        # Si no lo necesitamos, podemos dejarlo vacío o no implementarlo.
        print(f"Step {self.current_step} | State:\n{self.state}")



    def close(self):
        """
        Cierra el entorno.

        Se usa si se abren recursos externos (ventanas, archivos, conexiones). 
        En este entorno no hace nada.
        """
        # Tampoco es obligatorio.
        # Se usa si el entorno abre recursos externos (ventanas gráficas, archivos, conexiones, etc.).
        # Si no se habre nada, se puede dejarlo como pass.
        pass
