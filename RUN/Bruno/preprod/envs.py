import gymnasium as gym
from gymnasium import spaces
import numpy as np
# from rates import get_reward
import torch
# from utils import *


class APNetworkEnv(gym.Env):
    """
    Entorno de red inalámbrica para la asignación de canales y potencias en puntos de acceso (APs).
    
    Este entorno simula varios APs que deben seleccionar canal y nivel de potencia en cada paso,
    bajo la restricción de una potencia máxima relativa (Pmax) respecto a la máxima potencia a la 
    que pueden transmitir los APs (P0). 
    
    Compatible con Gymnasium y verificable con `stable_baselines3.common.env_checker.check_env`.

    ------------------------
    Espacio de observaciones
    ------------------------
    Dict con dos campos:

        - "H": matriz de tamaño (n_APs, n_APs)
            - Representa la ganancia de canal entre cada AP.
            - Tipo: np.ndarray, float32
        - "mu": vector de tamaño (n_APs,)
            - Multiplicadores de Lagrange o penalizaciones por potencia.
            - Tipo: np.ndarray, float32

    --------------------
    Espacio de acciones
    --------------------
    MultiDiscrete de tamaño n_APs. Cada acción de un AP puede ser:

        - 0: AP apagado
        - 1..(num_channels * n_power_levels): combinación de canal y nivel de potencia
          (el canal y la potencia elegida se decodifican mediante división y módulo usando n_power_levels)

        Ejemplo:
        Supongamos num_channels=2, n_power_levels=2. 
        Entonces cada acción puede ir de 0 a 4:

        Acción 0: AP apagado
        Acción 1: canal 1, potencia nivel 1
        Acción 2: canal 1, potencia nivel 2
        Acción 3: canal 2, potencia nivel 1
        Acción 4: canal 2, potencia nivel 2
    """

    metadata = {"render_modes": ["human"]}


    def __init__(self, n_APs=4, num_channels=3, P0=4, n_power_levels=3, 
                 power_levels_explicit=None, Pmax=0.7, max_steps=500, 
                 H_iterator=None, alpha=0.3, include_overlap=False):
        """
        Inicializa el entorno de red de APs.

        Parámetros
        ----------
        n_APs : int
            Número de puntos de acceso (APs) en la red.
        num_channels : int
            Número de canales disponibles para los APs.
        P0 : float
            Potencia máxima absoluta a la que pueden transmitir los APs.
        n_power_levels : int
            Número de niveles discretos de potencia que cada AP puede usar.
        power_levels_explicit : array-like, opcional
            Lista de niveles de potencia explícitos a usar en lugar de uniformes.
            Si se pasa, se ignoran los P0 y n_power_levels dados y pasan a ser calculados.
        Pmax : float
            Fracción de P0 que representa la potencia máxima permitida (ej. 0.7 = 70% de P0).
        max_steps : int
            Número máximo de pasos en un episodio.
        alpha : float
            Parámetro usado en el cálculo de reward (coeficiente de ponderación).
        """
        super().__init__()

        self.n_APs = n_APs
        self.num_channels = num_channels
        self.max_steps = max_steps
        self.alpha = alpha
        self.H_iterator = H_iterator

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
        # self.observation_space = spaces.Dict({
        #     "H": spaces.Box(low=0, high=np.inf, shape=(n_APs, n_APs), dtype=np.float32),
        #     "mu": spaces.Box(low=0, high=np.inf, shape=(n_APs,), dtype=np.float32)
        # })
        self.observation_space = spaces.Dict({
            "H":  spaces.Box(low=0, high=100, shape=(n_APs, n_APs), dtype=np.float32),
            "mu": spaces.Box(low=0, high=1, shape=(n_APs,), dtype=np.float32)
        })

        # --- Estado interno ---
        # self.state = None
        self.H = None
        #self.mu_power = np.zeros(self.n_APs, dtype=np.float32)
        self.mu_power = torch.ones(self.n_APs, dtype=torch.float32)
        self.current_step = 0
        self.power_history = []   # Para guardar el histórico de potencias y calcular el promedio luego
        self.include_overlap = include_overlap


    def reset(self, seed=None, options=None):
        """
        Resetea el entorno a un estado inicial.

        Parámetros
        ----------
        seed : int, opcional
            Semilla para reproducibilidad.
        options : dict, opcional
            Opciones adicionales (no usadas actualmente).

        Returns
        -------
        obs : dict
            Observación inicial {"H": np.ndarray, "mu": np.ndarray}.
        info : dict
            Diccionario vacío (compatibilidad Gymnasium).
        """

        super().reset(seed=seed)
        self.current_step = 0
        self.power_history = []

        # Genero canal inicial y mu inicial
        self.H = self._get_channel_matrix()
        self.mu_power = torch.zeros(self.n_APs, dtype=torch.float32)

        self.iterator_exhausted=False

        # Estado inicial aleatorio
        self.canales = np.random.randint(1, self.num_channels + 1, size=(self.n_APs, 1)).astype(np.float32)
        indices_pot = np.random.randint(0, self.n_power_levels, size=(self.n_APs, 1))
        self.potencias = self.power_levels[indices_pot].astype(np.float32)

        obs = {
            "H": self.H,
            "mu": self.mu_power if isinstance(self.mu_power, np.ndarray) else self.mu_power.detach().cpu().numpy()
        }
        info = {}

        return obs, info


    
    def step(self, action):
        """
        Aplica una acción y avanza un paso en el entorno.

        Parámetros
        ----------
        action : array-like
            Vector de acciones de tamaño n_APs. Cada elemento indica el canal y nivel de potencia elegido.


        Returns
        -------
        obs : dict
            Observación resultante {"H": np.ndarray, "mu": np.ndarray}.
        reward : float
            Recompensa obtenida en este paso.
        terminated : bool
            True si el episodio terminó por alguna condición de finalización.
        truncated : bool
            True si el episodio terminó por alcanzar `max_steps`.
        info : dict
            Información adicional, incluye promedio de potencias en "avg_powers".
        """

        self.current_step += 1
        action = np.array(action)   # ver si paso la accion de una o los logits

        if self.iterator_exhausted:
            terminated = True
            truncated = False
            reward = 0.0
            info = {"reason": "iterator_exhausted"}
            obs = {
                "H": -np.ones((self.n_APs, self.n_APs), dtype=np.float32),
                "mu": self.mu_power if isinstance(self.mu_power, np.ndarray) else self.mu_power.detach().cpu().numpy()
            }
            return obs, reward, terminated, truncated, info
        

        # Inicializamos estado nuevo con ceros
        # ¿necesario?
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


        reward = self._compute_reward(phi)

        # Actualizar canal y multiplicadores para el próximo paso       
        self.H = self._get_channel_matrix()        
        self.mu_power = self._update_mu()


        # terminated se usaría si, cumplida una condición, debe empezar el siguiente paso reseteando el state.
        terminated = False

        # Indica si el episodio terminó porque se alcanzó un limite impuesto (por ejemplo, max steps)
        truncated = self.current_step >= self.max_steps

        # info{} es un diccionario opcional que se usa para devolver información extra que no forma parte del estado, 
        # pero puede ser útil para debugging, logging o métricas.
        info = {"avg_powers": np.mean(self.power_history, axis=0)}


        obs = {
            "H": self.H,
            "mu": self.mu_power if isinstance(self.mu_power, np.ndarray) else self.mu_power.detach().cpu().numpy()
        }        

        return obs, reward, terminated, truncated, info
    


    def _compute_reward(self, phi):
        """
        Calcula la recompensa para un estado dado de potencia y canal (`phi`).

        Recompensa = suma de tasas (rate) - penalización por superar Pmax.

        Parámetros
        ----------
        phi : torch.Tensor
            Matriz de tamaño (n_APs, num_channels) con la potencia asignada por canal.

        Returns
        -------
        reward : float
            Valor de la recompensa para este paso.
        """
        # cambiar este mu, deberia usar mu_update creo
        # mu_power=0.5
        # Cambiar este sigma
        sigma=0.5

        # Rate general según la capacidad de Shannon
        all_rates = self._get_rates(phi, sigma)
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
        if self.H_iterator is None:
            raise ValueError("No hay iterador definido.")
        try:
            channel_matrix = next(self.H_iterator).astype(np.float32)
        except StopIteration:
            # Devuelvo matriz de -1
            channel_matrix = -np.ones((self.n_APs, self.n_APs), dtype=np.float32)
            # flag para avisar a step() que termine el episodio.
            self.iterator_exhausted = True
        return channel_matrix



    def _update_mu(self, eps=0.01):
        # ESTO ME LO DIO EL CHAT, ME DIJO QUE ES LA TIPICA.
        # COMPARAR CON LA DEL CHINO
        if len(self.power_history) == 0:
            return self.mu_power

        avg_power = np.mean(self.power_history, axis=0)
        self.mu_power = np.maximum(0.0, self.mu_power + eps * (avg_power - self.Pmax))
        return self.mu_power
    


    # def _get_rates(self, phi, sigma):
    #     """
    #     Calcula las tasas (rates) para el estado actual del entorno (sin batch).

    #     Parámetros
    #     ----------
    #     phi : torch.Tensor, shape (n_APs, num_channels)
    #         Potencia transmitida por cada AP en cada canal.
    #     sigma : float
    #         Ruido térmico o ruido de fondo.
        
    #     Returns
    #     -------
    #     rates : torch.Tensor, shape (n_APs,)
    #         Tasa de transmisión por AP.
    #     """
    #     H = torch.tensor(self.H, dtype=torch.float32)          # [n_APs, n_APs]
    #     sigma = torch.tensor(sigma, dtype=torch.float32)
    #     P0 = self.P0
    #     alpha = self.alpha

    #     n_APs, num_channels = phi.shape

    #     # Señal útil: potencia recibida de su propio canal
    #     diagH = torch.diagonal(H, dim1=0, dim2=1)              # [n_APs]
    #     signal = diagH.unsqueeze(-1) * phi                     # [n_APs, num_channels]

    #     # Interferencia intra-canal (misma frecuencia)
    # # interf_same = torch.matmul(H, phi) - signal            # [n_APs, num_channels]
    #     interf_base = torch.matmul(H, phi) - signal            # [n_APs, num_channels]
        
    #     # Factor de contención (p_mc / p0)
    #     coupling = phi / (P0 + 1e-12)

    #     # Interferencia modulada
    #     interf_same = interf_base * coupling
        
    #     interf_overlap=0

    #     if self.include_overlap:
    #         # Interferencia por canales solapados
    #         interf_overlap = torch.zeros_like(interf_same)
    #         for c in range(num_channels):
    #             if c > 0:
    #                 interf_overlap[:, c] += torch.matmul(H, phi[:, c-1])
    #             if c < num_channels - 1:
    #                 interf_overlap[:, c] += torch.matmul(H, phi[:, c+1])
    #         interf_overlap *= alpha

    #     # Denominador total
    #     denom = sigma + interf_same + interf_overlap

    #     # SINR
    #     snr = signal / denom

    #     # Tasa Shannon (log(1 + SINR))
    #     #rates = torch.sum(torch.log1p(snr), dim=-1)            # [n_APs]
    #     rates = torch.log1p(torch.sum(snr, dim=-1))
    #     print("shape de rates es:", rates.shape)
    #     return rates

    def _get_rates(self, phi, sigma, eps=1e-12):
        """
        Versión adaptada al paper manteniendo:
        - sin batch
        - canales solapados
        - suma de tasas por canal
        """
        H = torch.tensor(self.H, dtype=torch.float32)   # [n_APs, n_APs]
        sigma = torch.tensor(sigma, dtype=torch.float32)

        n_APs, num_channels = phi.shape

        # Señal útil
        diagH = torch.diagonal(H)                       # [n_APs]
        signal = diagH.unsqueeze(-1) * phi              # [n_APs, num_channels]

        # Interferencia intra-canal
        interf_same = torch.matmul(H, phi) - signal     # [n_APs, num_channels]

        # Interferencia por solapamiento
        interf_overlap = torch.zeros_like(interf_same)
        if self.include_overlap:
            for c in range(num_channels):
                if c > 0:
                    interf_overlap[:, c] += torch.matmul(H, phi[:, c-1])
                if c < num_channels - 1:
                    interf_overlap[:, c] += torch.matmul(H, phi[:, c+1])
            interf_overlap *= self.alpha

        # 🔹 NUEVO: coupling del paper
        coupling = phi / (self.P0 + eps)

        # 🔹 Interferencia modulada
        interf_total = interf_same + interf_overlap
        interf_mod = interf_total * coupling

        # SINR
        denom = sigma + interf_mod
        snr = signal / (denom + eps)

        # Rate
        rates = torch.sum(torch.log1p(snr), dim=-1)     # [n_APs]

        return rates

    def nuevo_get_rates(phi, H, sigma, eps=1e-12,p0=None):
        """
        TU VERSIÓN (del paper):
        Interferencia modulada por p_mc/p0
        
        Interpretación: "Rate efectivo considerando contención"
        """
        batch_size, num_links, num_channels = phi.shape
        
        diagH = torch.diagonal(H, dim1=1, dim2=2)  # [batch, num_links]
        signal = diagH.unsqueeze(-1) * phi  # [batch, num_links, num_channels]
        
        # Coupling factor: p_mc / p0
        coupling_factor = phi / (p0 + eps)
        
        # Interferencia base
        interference_base = torch.einsum('bij,bjc->bic', H.float(), phi.float())
        interference_base = interference_base - signal
        
        # Modular por coupling
        interference_modulated = interference_base * coupling_factor
        
        # SINR y rate
        sinr = signal / (sigma + interference_modulated + eps)
        rates = torch.log1p(torch.sum(sinr, dim=-1))
        
        return rates


    def render(self):
        """
        Muestra el estado actual del entorno en consola.

        Imprime para cada paso la matriz `phi` y la última recompensa obtenida.
        """
        # No es obligatorio, pero útil si se quiere visualizar lo que pasa (debugging o mostrar resultados).
        # Puede ser tan simple como un print o tan complejo como gráficos en tiempo real.
        # Si no lo necesitamos, podemos dejarlo vacío o no implementarlo.
        print(f"Step {self.current_step} \n Decisión:\n{self.last_phi} \n reward: {self.last_reward}")



    def close(self):
        """
        Cierra el entorno.

        No realiza ninguna acción en este entorno, presente solo por compatibilidad.
        """
        
        # Tampoco es obligatorio.
        # Se usa si el entorno abre recursos externos (ventanas gráficas, archivos, conexiones, etc.).
        pass


    
    
        
