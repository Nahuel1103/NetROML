import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from pathlib import Path

from processing.load_dataset import load_dataset
from processing.arrival_departure_model import ArrivalDepartureModel

class NetworkGraphEnv(gym.Env):
    """
    Entorno personalizado de Gym para simulación de RSSI en redes con observaciones de grafos.
    Incluye física para:
    - Clientes "pegajosos" (Sticky Clients): Umbral de re-asociación.
    - Interferencia de canal.
    - Cálculo de SINR y Tasa de bits.
    
    Refactorización:
    - Características de nodo específicas para APs y Clientes.
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
        
        # Cargar datos reales RSSI (usados para obtener valores entre clientes y APs)
        self.df = load_dataset(data_root, building_id)
        
        # Identificar todos los posibles APs y Clientes del dataset
        self.aps = sorted(self.df['mac_ap'].unique())
        all_clients = sorted(self.df['mac_cliente'].unique())
        
        self.ap_to_idx = {mac: i for i, mac in enumerate(self.aps)}
        self.all_client_macs = all_clients  # Store all possible client MACs
        
        self.num_aps = len(self.aps)
        print(f"APs: {self.num_aps}")
        print(f"Total possible clients in dataset: {len(all_clients)}")
        
        # Calcular el índice máximo de bloque para cada cliente
        client_block_counts = self.df.groupby('mac_cliente')['block_index'].max().to_dict()
        
        # Modelo de Arribo/Partida - simula qué clientes están activos
        self.max_timesteps = max_timesteps
        self.arrival_model = ArrivalDepartureModel(
            arrival_rate=arrival_rate,
            mean_duration=mean_duration,
            total_timesteps=max_timesteps,
            random_seed=random_seed,
            client_block_counts=client_block_counts
        )
        
        # Pre-generar todos los eventos de arribo/partida
        print("Generando eventos de arribo/partida...")
        self.arrival_model.simulate_all_events()
        stats = self.arrival_model.get_statistics()
        print(f"  Total simulated clients: {stats['total_clients']}")
        print(f"  Mean occupancy: {stats['mean_occupancy']:.1f} clients/timestep")
        print(f"  Max occupancy: {stats['max_occupancy']:.0f} clients")
        
        # Estado Dinámico - se actualizará en cada timestep
        self.current_step = 0
        self.active_clients = []  # Lista de eventos de clientes actualmente activos
        self.client_to_idx = {}  # Se reconstruirá en cada timestep
        self.num_clients = 0  # Se actualizará en cada timestep
        
        # Variables de estado de AP (tamaño fijo)
        self.ap_channels = np.zeros(self.num_aps, dtype=int)
        self.ap_powers = np.ones(self.num_aps, dtype=int) * 2  # Por defecto Alto
        
        # Conexiones de clientes (dinámico - redimensionado cada timestep)
        self.client_connections = np.array([], dtype=int)
        
        # Opciones de Configuración
        self.available_channels = [1, 6, 11]
        self.tx_powers_dbm = [10, 17, 23]  # Bajo, Medio, Alto
        
        self.action_space = spaces.MultiDiscrete([3, 3] * self.num_aps)
        
        self.observation_space = spaces.Dict({
            "num_nodes": spaces.Discrete(self.num_aps + len(all_clients) + 1),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Aleatorizar configuración inicial de APs
        self.ap_channels = np.random.randint(0, 3, size=self.num_aps)
        self.ap_powers = np.random.randint(0, 3, size=self.num_aps)
        
        # Obtener clientes activos del modelo de arribos en t=0
        self._update_active_clients()
        
        return self._get_observation(), {}
    
    def _update_active_clients(self):
        """Actualiza la lista de clientes activos basada en el modelo de arribos/partidas"""
        # Obtener eventos activos directamente - ahora contienen la MAC real como client_id
        self.active_clients = self.arrival_model.get_active_clients(self.current_step)
        self.num_clients = len(self.active_clients)
        
        # Reconstruir mapeo de clientes usando IDs de sesión únicos
        # Asignamos una ID sintética "session_i" al i-ésimo cliente activo para asegurar unicidad en el grafo
        self.client_to_idx = {f"session_{i}": i for i in range(self.num_clients)}
        
        # Redimensionar/reinicializar conexiones de clientes
        self.client_connections = np.full(self.num_clients, -1, dtype=int)

    def step(self, action):
        # 1. Aplicar Acciones
        action = action.reshape(self.num_aps, 2)
        self.ap_channels = action[:, 0]
        self.ap_powers = action[:, 1]
        
        # 2. Actualizar clientes activos según el modelo estocástico
        self._update_active_clients()
        
        # 3. Obtener datos RSSI para los clientes activos actualmente
        current_timeslot_data = self._get_current_snapshot()
        
        # 4. Física y Lógica Auxiliar
        self._update_client_connections(current_timeslot_data)
        total_rate, client_rates = self._calculate_network_rate(current_timeslot_data)
        
        # 5. Siguiente Paso
        self.current_step += 1
        terminated = self.current_step >= self.max_timesteps
        truncated = False
        
        observation = self._get_observation()
        info = {
            "mean_rate": np.mean(client_rates) if len(client_rates) > 0 else 0,
            "num_active_clients": self.num_clients,
            "num_arrivals": len(self.arrival_model.get_arrivals_at_timestep(self.current_step - 1)),
            "num_departures": len(self.arrival_model.get_departures_at_timestep(self.current_step - 1))
        }
        
        return observation, total_rate, terminated, truncated, info
    
    def _get_current_snapshot(self):
        """Obtiene datos RSSI del dataset para los clientes activos usando sus bloques asignados"""
        if self.num_clients == 0:
            return pd.DataFrame(columns=self.df.columns)
        
        snapshots = []
        for i, event in enumerate(self.active_clients):
            # event.client_id es la dirección MAC real
            mac = event.client_id
            block = event.block_index
            
            # Filtrar para este cliente y bloque específico
            client_data = self.df[
                (self.df['mac_cliente'] == mac) & 
                (self.df['block_index'] == block)
            ].copy()
            
            # Asignar ID de sesión única para que la lógica del entorno lo trate correctamente
            client_data['mac_cliente'] = f"session_{i}"
            
            snapshots.append(client_data)
            
        if not snapshots:
             return pd.DataFrame(columns=self.df.columns)
             
        return pd.concat(snapshots, ignore_index=True)

    def _update_client_connections(self, df_snapshot):
        # Lógica pegajosa (Sticky): igual que antes
        for client_mac, group in df_snapshot.groupby('mac_cliente'):
            if client_mac not in self.client_to_idx: continue
            c_idx = self.client_to_idx[client_mac]
            current_ap_idx = self.client_connections[c_idx]
            current_rssi = -100.0
            
            if current_ap_idx != -1:
                current_ap_mac = self.aps[current_ap_idx]
                row = group[group['mac_ap'] == current_ap_mac]
                if not row.empty:
                    current_rssi = float(row.iloc[0]['rssi'])
                else:
                    current_ap_idx = -1
                    current_rssi = -100.0

            best_ap_idx = -1
            best_rssi = -200.0
            
            for _, row in group.iterrows():
                ap_mac = row['mac_ap']
                if ap_mac not in self.ap_to_idx: continue
                ap_idx = self.ap_to_idx[ap_mac]
                p_idx = self.ap_powers[ap_idx]
                p_dbm = self.tx_powers_dbm[p_idx]
                adjusted_rssi = float(row['rssi']) - (23 - p_dbm)
                
                if adjusted_rssi > best_rssi:
                    best_rssi = adjusted_rssi
                    best_ap_idx = ap_idx
            
            threshold = -80.0
            if current_ap_idx == -1:
                if best_ap_idx != -1 and best_rssi > threshold:
                     self.client_connections[c_idx] = best_ap_idx
            else:
                if current_rssi < threshold:
                     if best_ap_idx != -1 and best_rssi > current_rssi:
                         self.client_connections[c_idx] = best_ap_idx

    def _calculate_network_rate(self, df_snapshot):
        # Lógica de cálculo de tasa con interferencia intra-AP e inter-AP
        total_rate = 0.0
        client_rates = []
        noise_mu_w = 10**(-90/10)
        
        signal_map = {}
        for _, row in df_snapshot.iterrows():
            if row['mac_cliente'] not in self.client_to_idx or row['mac_ap'] not in self.ap_to_idx: continue
            c_idx = self.client_to_idx[row['mac_cliente']]
            a_idx = self.ap_to_idx[row['mac_ap']]
            p_idx = self.ap_powers[a_idx]
            p_dbm = self.tx_powers_dbm[p_idx]
            rssi_adj = float(row['rssi']) - (23 - p_dbm)
            signal_map[(c_idx, a_idx)] = rssi_adj
            
        clients_per_ap = np.zeros(self.num_aps)
        for c_idx in range(self.num_clients):
            a_idx = self.client_connections[c_idx]
            if a_idx != -1: clients_per_ap[a_idx] += 1
                
        for c_idx in range(self.num_clients):
            a_idx = self.client_connections[c_idx]
            if a_idx == -1:
                client_rates.append(0.0)
                continue
            if (c_idx, a_idx) not in signal_map:
                client_rates.append(0.0)
                continue
                
            s_dbm = signal_map[(c_idx, a_idx)]
            s_mw = 10**(s_dbm/10)
            
            my_channel = self.ap_channels[a_idx]
            interference_mw = 0.0
            
            # Interferencia Intra-AP: Otros clientes conectados al mismo AP
            for other_c_idx in range(self.num_clients):
                if other_c_idx == c_idx: continue
                if self.client_connections[other_c_idx] == a_idx:  # Mismo AP
                    if (other_c_idx, a_idx) in signal_map:
                        i_dbm = signal_map[(other_c_idx, a_idx)]
                        interference_mw += 10**(i_dbm/10)
            
            # Interferencia Inter-AP: Otros APs en el mismo canal
            for other_a_idx in range(self.num_aps):
                if other_a_idx == a_idx: continue
                if self.ap_channels[other_a_idx] != my_channel: continue
                if (c_idx, other_a_idx) in signal_map:
                    i_dbm = signal_map[(c_idx, other_a_idx)]
                    interference_mw += 10**(i_dbm/10)
            
            sinr = s_mw / (noise_mu_w + interference_mw)
            rate = 20e6 * np.log2(1 + sinr)
            
            # Ya no dividimos por clients_per_ap porque la interferencia intra-AP
            # ya está modelada explícitamente en el cálculo de SINR
            rate_mbps = rate / 1e6
            total_rate += rate_mbps
            client_rates.append(rate_mbps)
            
        return total_rate, client_rates

    def _get_observation(self):
        # ---------------------------------------------------------
        # Construcción Refactorizada de Características
        # ---------------------------------------------------------
        
        # 1. Obtener snapshot de datos actual para clientes activos
        current_timeslot_data = self._get_current_snapshot()

        edge_index = []
        edge_attr = []
        
        # Normalizar RSSI para pesos de aristas (usando normalización estilo referencia si es posible, o solo positivo)
        # Referencia usaba transform_matrix/1e-3. Aquí usamos RSSI+100.
        
        for _, row in current_timeslot_data.iterrows():
            if row['mac_ap'] not in self.ap_to_idx or row['mac_cliente'] not in self.client_to_idx: continue
            a_idx = self.ap_to_idx[row['mac_ap']]
            c_idx = self.client_to_idx[row['mac_cliente']]
            
            src = a_idx
            dst = self.num_aps + c_idx
            rssi = float(row['rssi']) + 100.0
            
            edge_index.append([src, dst])
            edge_attr.append([rssi])
            edge_index.append([dst, src]) # Undirected/Bidirectional for propagation
            edge_attr.append([rssi])

        # 2. Características de Nodo
        # AP: [1.0, ID_Norm, Ch_Norm, Pwr_Norm]
        # Cliente: [-1.0, ID_Norm, -1.0, -1.0]
        
        x = []
        # Nodos AP
        for i in range(self.num_aps):
            id_norm = i / max(1, self.num_aps - 1)
            ch_norm = self.ap_channels[i] / 2.0
            p_norm = self.ap_powers[i] / 2.0
            x.append([1.0, id_norm, ch_norm, p_norm])
            
        # Nodos Cliente
        for i in range(self.num_clients):
            id_norm = i / max(1, self.num_clients - 1) if self.num_clients > 1 else 0.0
            x.append([-1.0, id_norm, -1.0, -1.0])
            
        x = torch.tensor(x, dtype=torch.float)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
