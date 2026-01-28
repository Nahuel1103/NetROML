import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import sys

# Allow importing from parent directory if running as script
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from processing.load_dataset import load_dataset
from processing.arrival_departure_model import ArrivalDepartureModel, build_client_block_counts
from processing.ap_action import APAction

class NetworkGraphEnv(gym.Env):
    """
    Entorno de Red WiFi controlado por GNN.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                data_root=Path("data"),
                building_id=990,
                max_timesteps=100,
                arrival_rate=3.0,
                mean_duration=15.0,
                random_seed=None):
        super(NetworkGraphEnv, self).__init__()
        
        # Cargar Dataset
        self.df = load_dataset(data_root, building_id)
        
        # APs y Clientes
        self.aps = sorted(self.df['mac_ap'].unique())
        self.clients = sorted(self.df['mac_cliente'].unique())
        
        self.ap_to_idx = {mac: i for i, mac in enumerate(self.aps)}

        self.num_aps = len(self.aps)
        self.total_clients = len(self.clients)
        
        
        print(f"APs: {self.num_aps}")
        print(f"Clientes: {self.total_clients}")
        
        # Máximo bloque disponible por cliente
        client_block_counts = build_client_block_counts(self.df)
        
        # Modelo de Arribo/Partida
        self.max_timesteps = max_timesteps
        self.arrival_departure_model = ArrivalDepartureModel(
            arrival_rate=arrival_rate,
            mean_duration=mean_duration,
            total_timesteps=max_timesteps,
            random_seed=random_seed,
            client_block_counts=client_block_counts
        )
        
        # Generar eventos
        print("Generando eventos de arribo/partida...")
        self.arrival_departure_model.simulate_all_events()
        
        # Estado Dinámico
        self.current_step = 0
        self.active_clients = []
        self.active_client_map = {} # session_id -> index
        self.num_active_clients = 0
        
        # Tensores de Estado AP
        self.ap_channels = torch.zeros(self.num_aps, dtype=torch.long)
        self.ap_powers_idx = torch.ones(self.num_aps, dtype=torch.long) * 2 
        
        # Conexiones [Num_Active_Clients]
        self.client_connections = torch.full((0,), -1, dtype=torch.long)
        
        # Constantes Físicas
        self.available_channels = torch.tensor([1, 6, 11], dtype=torch.long)
        self.tx_powers_dbm = torch.tensor([10.0, 17.0, 23.0], dtype=torch.float32)
        self.noise_floor_dbm = -90.0
        self.noise_mw = 10**(self.noise_floor_dbm / 10.0)
        
        # Matrices Pre-calculadas (se llenan en cada step)
        # Path Loss Matrix: [Num_Active_Clients, Num_APs]
        self.path_loss_matrix = None 
        
        # Espacio de acciones:
        # Acción plana: [ch_0, pwr_0, ch_1, pwr_1, ..., ch_N, pwr_N]
        # donde (ch_i, pwr_i) define la configuración del AP i
        self.action_space = spaces.MultiDiscrete(
            [len(self.available_channels), len(self.tx_powers_dbm)] * self.num_aps
        )        
        
        # Observation Space (Dict placeholder)
        self.observation_space = spaces.Dict({
            "num_active_clients": spaces.Discrete(10000),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Configuración Inicial Aleatoria
        self.ap_channels = torch.randint(0, len(self.available_channels), (self.num_aps,))
        self.ap_powers_idx = torch.randint(0, len(self.tx_powers_dbm), (self.num_aps,))
        
        # Obtener clientes activos del modelo de arribos en t=0
        self._update_active_clients()
        
        return self._get_observation(), {}
    
    def _update_active_clients(self):
        self.active_clients = self.arrival_departure_model.get_active_clients(self.current_step)
        self.num_active_clients = len(self.active_clients)
        
        # Mapeo de clientes activos para gestión interna
        self.active_client_map = {f"session_{i}": i for i in range(self.num_active_clients)}
        
        # Resetear conexiones (o mantener lógica sticky si se implementa persistencia completa)
        # Aquí reseteamos y dejamos que _update_client_connections decida
        self.client_connections = torch.full((self.num_active_clients,), -1, dtype=torch.long)
        
        # Construir matriz de Path Loss para el snapshot actual
        # Esto es lo más costoso, intentar hacerlo eficientemente
        current_snapshot = self._get_current_snapshot_df()
        
        # Matriz [Clients, APs] iniciada en muy bajo signal (alta pérdida)
        # Usamos RSSI directo como proxy de -PathLoss + TxPower_Ref
        # Asumimos que el RSSI en df es con TxPower ~23dBm (max) o similar.
        # Simplificación: Usamos el RSSI del Dataset como "Received Power con Max Tx Power"
        # Luego ajustamos por la diferencia de potencia real.
        
        self.base_rssi_matrix = torch.full((self.num_active_clients, self.num_aps), -120.0, dtype=torch.float32)
        
        if not current_snapshot.empty:
            # Vectorizar llenado
            # Necesitamos índices numéricos para APs y Clientes
            # current_snapshot tiene 'mac_cliente' como 'session_i' y 'mac_ap' real
            
            # Mapear mac_ap a idx
            current_snapshot['ap_idx'] = current_snapshot['mac_ap'].map(self.ap_to_idx)
            # Mapear session_id a idx
            current_snapshot['c_idx'] = current_snapshot['mac_cliente'].map(self.active_client_map)
            
            valid_rows = current_snapshot.dropna(subset=['ap_idx', 'c_idx'])
            
            if not valid_rows.empty:
                c_indices = torch.tensor(valid_rows['c_idx'].values, dtype=torch.long)
                a_indices = torch.tensor(valid_rows['ap_idx'].values, dtype=torch.long)
                rssi_vals = torch.tensor(valid_rows['rssi'].values, dtype=torch.float32)
                
                self.base_rssi_matrix[c_indices, a_indices] = rssi_vals

    def step(self, action):
        # 1. Aplicar Acción
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
            
        # Desempaquetar acción plana [ch0, p0, ch1, p1, ...]
        action = action.view(self.num_aps, 2)
        self.ap_channels = action[:, 0]
        self.ap_powers_idx = action[:, 1]
        
        # 2. Actualizar Clientes y Estado Físico
        self._update_active_clients()
        
        if self.num_active_clients > 0:
            # 3. Calcular Física Vectorizada
            rates, total_rate, fairness = self._calculate_metrics_vectorized()
        else:
            rates = torch.tensor([])
            total_rate = 0.0
            fairness = 1.0
            
        # 4. Observación
        obs = self._get_observation()
        
        # 5. Reward: Proportional Fairness -> Sum(log(rate + epsilon))
        # Rate en Mbps. epsilon para evitar log(0)
        epsilon_rate = 1e-6
        # Si rates está vacío, reward = 0
        if len(rates) > 0:
            reward = torch.sum(torch.log(rates + epsilon_rate)).item()
        else:
            reward = 0.0
            
        # 6. Info y Terminación
        self.current_step += 1
        terminated = self.current_step >= self.max_timesteps
        truncated = False
        
        info = {
            "mean_rate": rates.mean().item() if len(rates) > 0 else 0.0,
            "total_rate": total_rate,
            "fairness_index": fairness,
            "num_active_clients": self.num_active_clients
        }
        
        return obs, reward, terminated, truncated, info

    def _calculate_metrics_vectorized(self):
        """
        Calcula SINR y Rates usando tensores.
        Assumption: Downlink, Interfering APs only if they have active clients.
        """
        # 1. Ajustar RSSI basado en Potencia Actual
        # Base RSSI asume Max Power (idx=2, 23dBm). Ajustamos delta.
        # Tx Power actual
        current_tx_powers = self.tx_powers_dbm[self.ap_powers_idx] # [Num_APs]
        max_tx_power = self.tx_powers_dbm[-1] # 23.0
        
        # Delta: P_actual - P_max (será <= 0)
        power_delta = current_tx_powers - max_tx_power # [Num_APs]
        
        # Matriz de Potencia Recibida Actual [Clients, APs] (en dBm)
        # Broadcasting: [C, A] + [A] -> [C, A]
        current_rssi_matrix = self.base_rssi_matrix + power_delta.unsqueeze(0)
        current_rx_mw = 10**(current_rssi_matrix / 10.0)
        
        # 2. Determinar asociación (Sticky Logic simplificada: Max RSSI > Thr)
        # Ojo: La lógica sticky completa requiere estado previo. 
        # Si update_active_clients resetea, aquí hacemos asociación "greedy" por step.
        # Para consistencia con RL, greedy por step está bien por ahora.
        
        threshold_dbm = -80.0
        best_rssi, best_ap_indices = torch.max(current_rssi_matrix, dim=1) # [C], [C]
        
        connected_mask = best_rssi > threshold_dbm # [C]
        self.client_connections = torch.where(connected_mask, best_ap_indices, -1)
        
        # 3. Calcular Interferencia
        # Máscara de Clientes Conectados
        connected_clients_idx = torch.where(connected_mask)[0]
        if len(connected_clients_idx) == 0:
            return torch.tensor([]), 0.0, 0.0
            
        # APs Activos (tienen al menos 1 cliente)
        # active_aps_mask = torch.zeros(self.num_aps, dtype=torch.bool)
        # active_aps_mask[self.client_connections[connected_mask]] = True
        
        # Señal Útil por Cliente [C]
        # Handle -1 indices safely: replace -1 with 0 temporarily, then mask properly
        safe_connections = self.client_connections.clone()
        safe_connections[~connected_mask] = 0 # Dummy valid index
        
        signal_mw = torch.take_along_dim(current_rx_mw, safe_connections.unsqueeze(1), dim=1).squeeze() # [C]
        # Poner 0 a desconectados
        signal_mw[~connected_mask] = 0.0
        
        # Interferencia: Suma de P_rx de otros APs en el mismo canal
        # Expandir Canales AP [1, A] -> [C, A]
        ap_channels_expanded = self.ap_channels.unsqueeze(0).expand(self.num_active_clients, -1)
        
        # Canal del AP al que estoy conectado [C, 1]
        my_ap_conn = self.client_connections.clone()
        my_ap_conn[~connected_mask] = 0 # Dummy para evitar index error, luego enmascaramos
        my_channel = self.ap_channels[my_ap_conn].unsqueeze(1) # [C, 1]
        
        # Máscara Co-Canal: [C, A] (AP 'a' transmite en el mismo canal que mi AP)
        co_channel_mask = (ap_channels_expanded == my_channel)
        
        # Excluir mi propio AP de la interferencia
        # Crear indice de APs [1, A]
        ap_indices = torch.arange(self.num_aps).unsqueeze(0).expand(self.num_active_clients, -1)
        not_my_ap_mask = (ap_indices != self.client_connections.unsqueeze(1))
        
        # Máscara de APs Activos (Globalmente)
        # Solo APs que sirven a alguien contribuyen a interferencia
        aps_serving_users = torch.unique(self.client_connections[connected_mask])
        active_aps_mask_b = torch.zeros(self.num_aps, dtype=torch.bool)
        active_aps_mask_b[aps_serving_users] = True
        
        # Expandir Active APs [1, A]
        active_aps_expanded = active_aps_mask_b.unsqueeze(0).expand(self.num_active_clients, -1)
        
        # Máscara Final de Interferencia [C, A]
        # (Co-Canal) AND (No es mi AP) AND (AP está Activo)
        interference_mask = co_channel_mask & not_my_ap_mask & active_aps_expanded
        
        # Sumar Potencia de APs interferentes
        interference_mw_matrix = current_rx_mw * interference_mask.float()
        total_interference_mw = interference_mw_matrix.sum(dim=1) # [C]
        
        # SINR
        sinr = signal_mw / (self.noise_mw + total_interference_mw + 1e-12)
        
        # Shannon Capacity (Approx)
        bw_hz = 20e6
        rates = bw_hz * torch.log2(1 + sinr)
        rates_mbps = rates / 1e6
        rates_mbps[~connected_mask] = 0.0
        
        total_rate = rates_mbps.sum().item()
        
        # Jain's Fairness Index
        # (Sum x)^2 / (N * Sum x^2)
        # Solo sobre clientes activos totales intentando conectarse? O solo conectados?
        # Normalmente sobre todos los activos (para penalizar desconexiones)
        n = self.num_active_clients
        sum_r = rates_mbps.sum()
        sum_sq_r = (rates_mbps ** 2).sum()
        fairness = (sum_r ** 2) / (n * sum_sq_r + 1e-9)
        
        return rates_mbps, total_rate, fairness.item()

    def _get_observation(self):
        """
        Genera un objeto HeteroData.
        """
        data = HeteroData()
        
        # -------------------
        # NODES
        # -------------------
        # AP Features: [ID_Norm, Ch_Norm, Pwr_Norm, Active_Flag]
        # Active_Flag: Indica si el AP tiene usuarios conectados
        is_active = torch.zeros(self.num_aps, dtype=torch.float)
        if hasattr(self, 'client_connections'):
             # Marcar activos
             active_indices = torch.unique(self.client_connections[self.client_connections != -1])
             is_active[active_indices] = 1.0
             
        ap_feats = torch.stack([
            torch.linspace(0, 1, self.num_aps),
            self.ap_channels.float() / 11.0, # Normalizar canal aprox
            self.ap_powers_idx.float() / 2.0,
            is_active
        ], dim=1)
        
        data['ap'].x = ap_feats
        
        # Client Features: [ID_Norm (dummy), 0, 0, 0] (Placeholder simple)
        # Podríamos poner QoS required si tuviéramos
        if self.num_active_clients > 0:
            c_feats = torch.zeros((self.num_active_clients, 4), dtype=torch.float)
            data['client'].x = c_feats # Features minimalistas
        else:
            data['client'].x = torch.empty((0, 4), dtype=torch.float)

        # -------------------
        # EDGES
        # -------------------
        # 'ap' -> 'connects' -> 'client' (Potencial Conexión / RSSI real)
        # Usamos base_rssi_matrix > Threshold mínimo de visibilidad (-100 dBm)
        if self.num_active_clients > 0:
            visibility_mask = self.base_rssi_matrix > -100.0
            src_c, dst_a = torch.where(visibility_mask) # Ojo: matrix es [C, A]
            
            # HeteroData edge direction: AP -> Client (Downlink)
            edge_index_ap_client = torch.stack([dst_a, src_c], dim=0)
            
            # Edge Attr: [RSSI_Norm]
            rssi_vals = self.base_rssi_matrix[src_c, dst_a]
            edge_attr = (rssi_vals + 100.0) / 50.0 # Norm approx [0, 1]
            
            data['ap', 'connects', 'client'].edge_index = edge_index_ap_client
            data['ap', 'connects', 'client'].edge_attr = edge_attr.unsqueeze(1)
            
            # 'client' -> 'connected_to' -> 'ap' (Estado actual de conexión)
            # Para que el GNN sepa quién está conectado a quién
            connected_indices = torch.where(self.client_connections != -1)[0]
            if len(connected_indices) > 0:
                  src_c_conn = connected_indices
                  dst_a_conn = self.client_connections[connected_indices]
                  data['client', 'connected_to', 'ap'].edge_index = torch.stack([src_c_conn, dst_a_conn], dim=0)
            else:
                  data['client', 'connected_to', 'ap'].edge_index = torch.empty((2, 0), dtype=torch.long)
                  
        else:
             data['ap', 'connects', 'client'].edge_index = torch.empty((2, 0), dtype=torch.long)
             data['ap', 'connects', 'client'].edge_attr = torch.empty((0, 1), dtype=torch.float)
             data['client', 'connected_to', 'ap'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 'ap' -> 'interferes' -> 'client'
        # Podríamos añadir esto explícitamente, o dejar que el GNN lo infiera via AP->Client edges + AP Features (Channel)
        # Por eficiencia, dejamos que el GNN use 'connects' y filtremos por atención si es necesario.
        # Pero mi plan decía explícitamente Interference edges.
        # Vamos a reutilizar 'connects' como 'signal path' que transporta tanto señal útil como interferencia.
        # GATv2 en el edge AP->Client sabrá si es útil o interferencia mirando los canales de los nodos AP.
        
        return data

    def _get_current_snapshot_df(self):
        """
        Reconstruye el DataFrame para los clientes activos en el timestep actual.
        """
        if self.num_active_clients == 0:
            return pd.DataFrame()

        snapshots = []
        # Optimización: filtrar el df principal una sola vez si es posible, 
        # pero como cada cliente tiene bloque aleatorio, hay que iterar.
        
        # Precargar datos necesarios del DF principal
        # Esto podría ser lento si DF es gigante.
        
        for i, event in enumerate(self.active_clients):
            session_id = f"session_{i}"
            mac_real = event.client_id
            
            # Obtener bloque
            block = self.arrival_departure_model.get_block_index_at_timestep(event, self.current_step)
            
            # Filtrar (pandas optimizado)
            # Asumimos que self.df tiene índice o es rápido.
            # Podemos optimizar haciendo un multi-query o mask grande.
            
            client_data = self.df[
                (self.df['mac_cliente'] == mac_real) &
                (self.df['block_index'] == block)
            ].copy()
            
            if not client_data.empty:
                client_data['mac_cliente'] = session_id # Override con ID de sesión único
                snapshots.append(client_data)

        if not snapshots:
            return pd.DataFrame()

        return pd.concat(snapshots, ignore_index=True)

