import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rates import get_reward
import torch
from utils import *


class APNetworkEnv(gym.Env):
    """
    Entorno de red inalámbrica para la asignación de canales y potencias en puntos de acceso (APs).
    
    Este entorno simula varios APs que deben seleccionar canal y nivel de potencia en cada paso,
    bajo la restricción de una potencia máxima relativa (Pmax) a potencia máxima a la que pueden
    transmitir los APs (P0). 
    
    Compatible con Gymnasium.

    ------------------------
    Espacio de observaciones
    ------------------------
    Representa el estado actual de cada AP. Depende del parámetro `flatten_obs`:

    - Si flatten_obs=False: matriz de tamaño (n_APs, 2)
        - columna 0: canal asignado al AP (1..num_channels)
        - columna 1: potencia actual del AP (valor float entre 0 y P0)

    - Si flatten_obs=True: vector 1D de tamaño n_APs*2
        - orden: [canal_1, potencia_1, canal_2, potencia_2, ..., canal_n, potencia_n]

    --------------------
    Espacio de acciones
    --------------------
    Vector de tamaño n_APs (MultiDiscrete), donde cada elemento representa la acción de un AP:

        - 0: AP apagado
        - 1..(num_channels * n_power_levels): combinación de canal y nivel de potencia
          (el canal y la potencia elegida se decodifican mediante división y módulo usando n_power_levels)

        Ejemplo:
        Supongamos n_APs=2, num_channels=2, n_power_levels=2. 
        Entonces cada acción puede ir de 0 a 4:

        Acción 0: AP apagado
        Acción 1: canal 1, potencia nivel 1
        Acción 2: canal 1, potencia nivel 2
        Acción 3: canal 2, potencia nivel 1
        Acción 4: canal 2, potencia nivel 2
    """

    metadata = {"render_modes": ["human"]}


    def __init__(self, n_APs=5, num_channels=3, P0=4, n_power_levels=3, 
                 power_levels_explicit=None, Pmax=0.7, max_steps=500, alpha=0.3):
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
        self.max_steps = max_steps
        self.alpha = alpha

        if power_levels_explicit is not None:
            self.power_levels=power_levels_explicit
            self.n_power_levels=len(self.power_levels)
            self.P0 = max(self.power_levels)

        else:
            self.n_power_levels=n_power_levels
            self.P0 = P0
            self.power_levels = (np.arange(1, self.n_power_levels + 1) / self.n_power_levels) * self.P0

        self.Pmax = P0*Pmax


        # --- Espacio de acciones ---
        # Cada AP tiene (num_channels * n_power_levels + 1) posibles acciones (incluye apagado)
        self.action_space = spaces.MultiDiscrete(
            [(self.num_channels * self.n_power_levels + 1)] * self.n_APs
        )
        

        # --- Espacio de observaciones ---
        self.observation_space = spaces.Dict({
            "H": spaces.Box(low=0, high=np.inf, shape=(n_APs, n_APs), dtype=np.float32),
            "mu": spaces.Box(low=0, high=np.inf, shape=(n_APs,), dtype=np.float32)
        })

        # --- Estado interno ---
        # self.state = None
        self.H = None
        self.mu_power = np.zeros(self.n_APs, dtype=np.float32)
        self.current_step = 0
        self.power_history = []   # Para guardar el histórico de potencias y calcular el promedio luego



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
        self.power_history = []

        # Genero canal inicial y mu inicial
        self.H = self._get_channel_matrix()
        self.mu_power = np.zeros(self.n_APs, dtype=np.float32)


        # Estado inicial aleatorio
        self.canales = np.random.randint(1, self.num_channels + 1, size=(self.n_APs, 1)).astype(np.float32)
        indices_pot = np.random.randint(0, self.n_power_levels, size=(self.n_APs, 1))
        self.potencias = self.power_levels[indices_pot].astype(np.float32)

        obs = {"H": self.H, "mu": self.mu_power}
        info = {}

        return obs, info


    
    def step(self, action):
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
        action = np.array(action)   # ver si paso la accion de una o los logits

        # Actualizar canal y multiplicadores
        ##### ¿ESTO TENDRIA QUE HACERLO AHORA O AL FINAL?
        self.H = self._get_channel_matrix()
        ##### el mu se actualiza asi o lo hace la red?
        self.mu_power = self._update_mu()

        # Inicializamos estado nuevo con ceros
        new_state = np.zeros((self.n_APs, 2), dtype=np.float32)

        # Máscara con APs activos (no apagados)
        active_mask = action > 0

        # Ajustamos acciones (para que 0 quede apagado y el resto arranque desde 0)
        a_adj = action[active_mask] - 1

        # Hallo que canal usan (solo para los activos)
        can = a_adj // self.n_power_levels + 1

        # Vemos que potencia usan (solo para los activos)
        # indices_asignacion_potencias = a_adj % self.n_power_levels        # potencias que eligió cada uno (indices en el array de posibles)
        # potencias = self.power_levels[indices_asignacion_potencias].astype(np.float32)        # potencas reales pero que eliguó cada uno
        indices_pot = a_adj % self.n_power_levels
        pot = self.power_levels[indices_pot].astype(np.float32)


        # Asignamos todo de una sola vez
        new_state[active_mask, 0] = can
        new_state[active_mask, 1] = pot

        self.canales = new_state[:, 0]
        self.potencias = new_state[:, 1]


        # Construyo phi
        phi = torch.zeros((self.n_APs, self.num_channels), dtype=torch.float32)
        canales_idx = torch.tensor(self.canales - 1, dtype=torch.long)
        phi[torch.arange(self.n_APs), canales_idx] = torch.tensor(self.potencias, dtype=torch.float32)


        # Guardar todas las potencias del paso actual en el histórico
        self.power_history.append(self.potencias.copy())


        reward = self._get_reward(phi)
        

        # terminated se usaría si, cumplida una condición, debe empezar el siguiente paso reseteando el state.
        terminated = False

        # Indica si el episodio terminó porque se alcanzó un limite impuesto (por ejemplo, max steps)
        truncated = self.current_step >= self.max_steps

        # info{} es un diccionario opcional que se usa para devolver información extra que no forma parte del estado, 
        # pero puede ser útil para debugging, logging o métricas.
        info = {"avg_powers": np.mean(self.power_history, axis=0)}


        obs = {"H": self.H, "mu": self.mu_power}

        return obs, reward, terminated, truncated, info
    


    def _get_reward(self, phi):
        # cambiar este mu, deberia usar mu_update creo
        # mu_power=0.5
        # Cambiar este sigma
        sigma=0.5

        # Rate general según la capacidad de Shannon
        all_rates = nuevo_get_rates(phi, self.H, sigma, self.P0, self.alpha)
        # rate = torch.sum(all_rates, dim=1) #puede que este sum este mal o haya que hacer otro sum mas
        rate = torch.sum(all_rates)  # suma total de tasas

        # Power constraint
        avg_power = torch.tensor(np.mean(self.power_history, axis=0))   # [nAPs,1]
        power_penalty = self.mu_power*(avg_power - self.Pmax)   # [nAPs,1]
        power_penalty = torch.sum(torch.clamp(power_penalty, min=0.0))    # solo penalizo si es positivo
        
        reward = rate - power_penalty

        # Solo para render
        self.last_reward = float(reward.item())  
        self.last_phi = phi.clone().detach()

        return float(reward.item())

    
    
    def _get_channel_matrix(self):
        # ESTO NO ES LO QUE QUEREMOS, PERO ES PARA VER SI ANDA AUNQUE SEA DESPUES
        H = np.random.rayleigh(scale=1.0, size=(self.n_APs, self.n_APs)).astype(np.float32)
        np.fill_diagonal(H, 1.0)  # Ganancia propia más fuerte
        return H
        


    def _update_mu(self, lr=0.01):
        # ESTO ME LO DIO EL CHAT, ME DIJO QUE ES LA TIPICA.
        # COMPARAR CON LA DEL CHINO
        if len(self.power_history) == 0:
            return self.mu_power

        avg_power = np.mean(self.power_history, axis=0)
        self.mu_power = np.maximum(0.0, self.mu_power + lr * (avg_power - self.Pmax))
        return self.mu_power


    def render(self):
        """
        Muestra el estado actual del entorno.

        Para este entorno, simplemente imprime el estado con canales y potencias de cada AP.
        """
        # No es obligatorio, pero útil si se quiere visualizar lo que pasa (debugging o mostrar resultados).
        # Puede ser tan simple como un print o tan complejo como gráficos en tiempo real.
        # Si no lo necesitamos, podemos dejarlo vacío o no implementarlo.
        print(f"Step {self.current_step} | Decisión:\n{self.last_phi} | reward: {self.last_reward}")



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


    
    
        
