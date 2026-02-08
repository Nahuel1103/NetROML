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
                random_seed=None,
                debug=False):
        super(NetworkGraphEnv, self).__init__()

        # Debug flag
        self.debug = debug
        
        # Cargar Dataset
        self.df = load_dataset(data_root, building_id)
        
        # APs y Clientes
        self.aps = sorted(self.df['mac_ap'].unique())
        self.clients = sorted(self.df['mac_cliente'].unique())
        
        self.ap_to_idx = {mac: i for i, mac in enumerate(self.aps)}
        self.num_aps = len(self.aps)
        self.total_clients = len(self.clients)
        
        print(f"APs: {self.num_aps}")
        print(f"Total Clients: {self.total_clients}")
        
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
        
        # Sticky Logic State
        self.session_connection_map = {} # session_id -> ap_idx
        
        # Estado AP
        self.ap_channels_idx = torch.zeros(self.num_aps, dtype=torch.long)
        self.ap_powers_idx = torch.zeros(self.num_aps, dtype=torch.long)
        
        # Conexiones [Num_Active_Clients]
        self.client_connections = torch.full((0,), -1, dtype=torch.long)
        
        # Constantes Físicas
        self.available_channels = torch.tensor([1, 6, 11], dtype=torch.long)
        self.tx_powers_dbm = torch.tensor([20, 17, 14, 11, 8], dtype=torch.long)
        
        self.p_tx_data_dbm = 14.0
        self.noise_floor_dbm = -90.0
        self.noise_mw = 10**(self.noise_floor_dbm / 10.0)
        
        # Thresholds
        self.threshold_lost_dbm = -90.0
        self.delta_sticky_db = 15.0
        
        # Matrices Pre-calculadas
        self.base_gain_matrix = None 
        
        # Espacio de acciones:
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
        self.ap_channels_idx = torch.randint(0, len(self.available_channels), (self.num_aps,))
        self.ap_powers_idx = torch.randint(0, len(self.tx_powers_dbm), (self.num_aps,))
        
        self.session_connection_map = {}
        self._update_active_clients()
        # self._perform_association_sticky() # Called in first step or manually if needed
        
        return self._get_observation(), {}
    
    def _update_active_clients(self):
        self.active_clients = self.arrival_departure_model.get_active_clients(self.current_step)
        self.num_active_clients = len(self.active_clients)
        
        # Mapeo de clientes activos para gestión interna
        self.active_client_map = {f"session_{i}": i for i in range(self.num_active_clients)}
        
        # Cleanup disconnected sessions from map
        active_session_ids = set(self.active_client_map.keys())
        keys_to_remove = [k for k in self.session_connection_map if k not in active_session_ids]
        for k in keys_to_remove:
            del self.session_connection_map[k]
        
        # Reset connections (will be recalculated by association logic)
        self.client_connections = torch.full((self.num_active_clients,), -1, dtype=torch.long)
        
        # Construir matriz de Ganancia (Gain)
        current_snapshot = self._get_current_snapshot_df()
        
        # Init with very low Gain
        self.base_gain_matrix = torch.full((self.num_active_clients, self.num_aps), -200.0, dtype=torch.float32)
        
        if not current_snapshot.empty:
            current_snapshot['ap_idx'] = current_snapshot['mac_ap'].map(self.ap_to_idx)
            current_snapshot['c_idx'] = current_snapshot['mac_cliente'].map(self.active_client_map)
            
            valid_rows = current_snapshot.dropna(subset=['ap_idx', 'c_idx'])
            
            if not valid_rows.empty:
                c_indices = torch.tensor(valid_rows['c_idx'].values, dtype=torch.long)
                a_indices = torch.tensor(valid_rows['ap_idx'].values, dtype=torch.long)
                rssi_vals = torch.tensor(valid_rows['rssi'].values, dtype=torch.float32)
                
                # Gain = RSSI - P_tx_data
                gain_vals = rssi_vals - self.p_tx_data_dbm
                self.base_gain_matrix[c_indices, a_indices] = gain_vals

    def step(self, action):
        # 1. Aplicar Acción
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
            
        action = action.view(self.num_aps, 2)
        self.ap_channels_idx = action[:, 0]
        self.ap_powers_idx = action[:, 1]
        
        # 2. Actualizar Clientes y Estado Físico
        self._update_active_clients()
        
        # 3. Asociación (Sticky) y Cálculo de Métricas
        if self.num_active_clients > 0:
            self._perform_association_sticky()
            rates, total_rate, fairness = self._calculate_metrics()
        else:
            self.client_connections = torch.tensor([], dtype=torch.long)
            rates = torch.tensor([])
            total_rate = 0.0
            fairness = 1.0
            
        # 4. Observación
        obs = self._get_observation()
        
        # 5. Reward
        epsilon_rate = 1e-6
        alpha_throughput = 0.7
        alpha_fairness = 0.3

        if len(rates) > 0:
            throughput_reward = torch.sum(torch.log(rates.mean() + epsilon_rate))
            fairness_penalty = fairness  # ya está entre 0-1
            reward = alpha_throughput * throughput_reward + alpha_fairness * fairness_penalty
        else:
            reward = -10.0  # Penalización por no conectar a nadie
            
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

    def _perform_association_sticky(self):
        """
        Determines client connection based on Sticky Logic.
        """
        current_tx_powers = self.tx_powers_dbm[self.ap_powers_idx]
        current_rx_matrix = self.base_gain_matrix + current_tx_powers.unsqueeze(0)
        
        new_client_connections = torch.full((self.num_active_clients,), -1, dtype=torch.long)
        
        for i in range(self.num_active_clients):
            session_id = f"session_{i}"
            
            rx_powers = current_rx_matrix[i]
            best_ap_idx = torch.argmax(rx_powers).item()
            best_rx_val = rx_powers[best_ap_idx].item()
            
            prev_ap_idx = self.session_connection_map.get(session_id, -1)
            conn_ap = -1
            
            if prev_ap_idx != -1:
                # Check Sticky
                current_link_rx = rx_powers[prev_ap_idx].item()
                
                # Lost?
                if current_link_rx < self.threshold_lost_dbm:
                    if best_rx_val > self.threshold_lost_dbm:
                        conn_ap = best_ap_idx
                else:
                    # Better?
                    if best_rx_val > current_link_rx + self.delta_sticky_db:
                        conn_ap = best_ap_idx
                    else:
                        conn_ap = prev_ap_idx
            else:
                # New
                if best_rx_val > self.threshold_lost_dbm:
                    conn_ap = best_ap_idx
            
            if conn_ap != -1:
                new_client_connections[i] = conn_ap
                self.session_connection_map[session_id] = conn_ap
            else:
                if session_id in self.session_connection_map:
                    del self.session_connection_map[session_id]
        
        self.client_connections = new_client_connections

    def _calculate_metrics(self):
        current_tx_powers = self.tx_powers_dbm[self.ap_powers_idx]
        current_rx_matrix = self.base_gain_matrix + current_tx_powers.unsqueeze(0)
        current_rx_mw = 10**(current_rx_matrix / 10.0)
        
        connected_mask = (self.client_connections != -1)
        if not connected_mask.any():
            return torch.tensor([]), 0.0, 1.0
            
        # Signal
        safe_connections = self.client_connections.clone()
        safe_connections[~connected_mask] = 0
        signal_mw = torch.take_along_dim(current_rx_mw, safe_connections.unsqueeze(1), dim=1).squeeze()
        signal_mw[~connected_mask] = 0.0
        
        # Interference Calculation
        ap_channels = self.ap_channels_idx
        ap_channels_exp = ap_channels.unsqueeze(0).expand(self.num_active_clients, -1)
        
        my_conn = self.client_connections.clone()
        my_conn[~connected_mask] = 0
        my_channels = ap_channels[my_conn].unsqueeze(1)
        
        co_channel_mask = (ap_channels_exp == my_channels)
        
        ap_indices = torch.arange(self.num_aps).unsqueeze(0).expand(self.num_active_clients, -1)
        not_my_ap_mask = (ap_indices != self.client_connections.unsqueeze(1))
        
        active_aps = torch.unique(self.client_connections[connected_mask])
        active_aps_mask = torch.zeros(self.num_aps, dtype=torch.bool)
        active_aps_mask[active_aps] = True
        active_aps_exp = active_aps_mask.unsqueeze(0).expand(self.num_active_clients, -1)
        
        interference_mask = co_channel_mask & not_my_ap_mask & active_aps_exp
        
        interference_mw_cont = current_rx_mw * interference_mask.float()
        total_interference_mw = interference_mw_cont.sum(dim=1)
        
        sinr = signal_mw / (self.noise_mw + total_interference_mw + 1e-12)
        
        bw_hz = 20e6
        rates = bw_hz * torch.log2(1 + sinr)
        rates_mbps = rates / 1e6
        rates_mbps[~connected_mask] = 0.0
        
        total_rate = rates_mbps.sum().item()
        
        n = self.num_active_clients
        sum_r = rates_mbps.sum()
        sum_sq_r = (rates_mbps ** 2).sum()
        fairness = (sum_r ** 2) / (n * sum_sq_r + 1e-12)
        
        return rates_mbps, total_rate, fairness.item()

    def _get_observation(self):
        data = HeteroData()
        
        # -------------------
        # NODES
        # -------------------
        is_active = torch.zeros(self.num_aps, dtype=torch.float)
        if hasattr(self, 'client_connections'):
             active_indices = torch.unique(self.client_connections[self.client_connections != -1])
             is_active[active_indices] = 1.0
             
        ap_feats = torch.stack([
            torch.linspace(0, 1, self.num_aps),
            self.ap_channels_idx.float() / 11.0, 
            self.ap_powers_idx.float() / 2.0,
            is_active
        ], dim=1)
        
        data['ap'].x = ap_feats
        
        # Client Features: [Connected_RSSI_Norm, Session_Duration_Norm, Connected_Flag, 0]
        if self.num_active_clients > 0:
            # 1. Connected RSSI
            # RSSI = Gain + Tx Power
            # Si no conectado, usar valor muy bajo (e.g. -120 dBm)
            
            # Tx Powers actuales
            current_tx_powers = self.tx_powers_dbm[self.ap_powers_idx]
            
            # Gain actual [C, A]
            current_gain_matrix = self.base_gain_matrix
            
            # Select Gain of connected AP
            connected_indices = self.client_connections.clone()
            mask_connected = (connected_indices != -1)
            
            # Placeholder for RSSI
            client_rssi = torch.full((self.num_active_clients,), -120.0, dtype=torch.float32)
            
            if mask_connected.any():
                # Indices validos
                valid_conn_aps = connected_indices[mask_connected]
                valid_clients_idx = torch.where(mask_connected)[0]
                
                # Gains para conexiones activas
                gains = current_gain_matrix[valid_clients_idx, valid_conn_aps]
                
                # Tx Power de esos APs
                tx_pwrs = current_tx_powers[valid_conn_aps]
                
                client_rssi[mask_connected] = gains + tx_pwrs
            
            # Normalize RSSI: map approx [-100, -30] to [0, 1]
            # (RSSI + 100) / 70
            client_rssi_norm = (client_rssi + 100.0) / 70.0
            
            # 2. Session Duration
            durations = torch.zeros(self.num_active_clients, dtype=torch.float32)
            for i, event in enumerate(self.active_clients):
                # event.arrival_time es absoluto? Checks ArrivalDepartureModel
                # Asumimos event.arrival_time es el step de inicio
                d = self.current_step - event.arrival_time
                durations[i] = d
            
            # Normalize Duration: / max_timesteps (approx)
            duration_norm = durations / self.max_timesteps
            
            # 3. Connected Flag
            connected_flag = mask_connected.float()
            
            # Stack features
            # [RSSI, Duration, Connected, 0]
            c_feats = torch.stack([
                client_rssi_norm,
                duration_norm,
                connected_flag,
                torch.zeros(self.num_active_clients)
            ], dim=1)
            
            data['client'].x = c_feats
        else:
            data['client'].x = torch.empty((0, 4), dtype=torch.float)

        # -------------------
        # EDGES
        # -------------------
        if self.num_active_clients > 0:
            # =====================================================
            # 1. ARISTAS DOWNLINK: AP → Cliente
            # Representa: Visibilidad RF (qué APs puede ver cada cliente)
            # =====================================================

            # Use Base Gain > -100 threshold for graph edges
            # Gain values typically -50 to -90. -100 as cutoff.
            visibility_mask = self.base_gain_matrix > -100.0
            client_idx, ap_idx = torch.where(visibility_mask)
            
            # edge_index = [source_nodes, target_nodes]
            # Para AP → Cliente: source=AP, target=Cliente
            edge_index_downlink = torch.stack([ap_idx, client_idx], dim=0)
            
            # Edge Attr: [Gain_Norm]
            # Normalization: Gain usually [-100, -30]. +100 / 70 => [0, 1] approx
            gain_vals = self.base_gain_matrix[client_idx, ap_idx]
            edge_attr = (gain_vals + 100.0) / 70.0 
            
            data['ap', 'connects', 'client'].edge_index = edge_index_downlink
            data['ap', 'connects', 'client'].edge_attr = edge_attr.unsqueeze(1)
            
            # =====================================================
            # 2. ARISTAS UPLINK: Cliente → AP
            # Representa: Conexión activa (carga en el AP)
            # =====================================================
            connected_client_indices = torch.where(self.client_connections != -1)[0]
            if len(connected_client_indices) > 0:
                connected_ap_indices = self.client_connections[connected_client_indices]
                
                # Para Cliente → AP: source=Cliente, target=AP
                edge_index_uplink = torch.stack([connected_client_indices, connected_ap_indices], dim=0)
                
                data['client', 'connected_to', 'ap'].edge_index = edge_index_uplink
            else:
                data['client', 'connected_to', 'ap'].edge_index = torch.empty((2, 0), dtype=torch.long)
                  
        else:
            # Sin clientes activos: grafos vacíos
            data['ap', 'connects', 'client'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['ap', 'connects', 'client'].edge_attr = torch.empty((0, 1), dtype=torch.float)
            data['client', 'connected_to', 'ap'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Solo validar si está en modo debug
        if self.debug:
            self._validate_graph(data)
        
        return data


    def _validate_graph(self, data):
        """Validaciones de integridad del grafo."""
        # Validar estructura
        assert data['ap', 'connects', 'client'].edge_index.shape[0] == 2
        
        # Validar rangos
        if data['ap', 'connects', 'client'].edge_index.shape[1] > 0:
            max_ap_idx = data['ap', 'connects', 'client'].edge_index[0].max().item()
            max_client_idx = data['ap', 'connects', 'client'].edge_index[1].max().item()
            
            assert max_ap_idx < self.num_aps, \
                f"AP index {max_ap_idx} >= num_aps {self.num_aps}"
            assert max_client_idx < self.num_active_clients, \
                f"Client index {max_client_idx} >= num_active_clients {self.num_active_clients}"


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
            
            # Filtrar 
        
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
