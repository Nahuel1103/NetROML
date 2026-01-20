import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

class NetworkGraphEnv(gym.Env):
    """
    Custom Gym Environment for Network RSSI Simulation with Graph Observations.
    Includes physics for:
    - Sticky Clients (Re-association threshold)
    - Channel Interference
    - SINR & Rate Calculation
    
    Refactoring:
    - Specific Node Features for APs and Clients.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_file):
        super(NetworkGraphEnv, self).__init__()
        
        self.df = pd.read_csv(data_file)
        self.timeslots = sorted(self.df['timeslot'].unique())
        self.current_step = 0
        
        # Identify Nodes
        self.aps = sorted(self.df['mac_ap'].unique())
        self.clients = sorted(self.df['mac_cliente'].unique())
        
        self.ap_to_idx = {mac: i for i, mac in enumerate(self.aps)}
        self.client_to_idx = {mac: i for i, mac in enumerate(self.clients)}
        
        self.num_aps = len(self.aps)
        self.num_clients = len(self.clients)
        
        # State Variables
        self.ap_channels = np.zeros(self.num_aps, dtype=int)
        self.ap_powers = np.ones(self.num_aps, dtype=int) * 2 # Default High
        self.client_connections = np.full(self.num_clients, -1, dtype=int)
        
        # Config options
        self.available_channels = [1, 6, 11]
        self.tx_powers_dbm = [10, 17, 23] # Low, Med, High
        
        self.action_space = spaces.MultiDiscrete([3, 3] * self.num_aps)
        
        self.observation_space = spaces.Dict({
            "num_nodes": spaces.Discrete(self.num_aps + self.num_clients + 1),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Randomize initial AP config
        self.ap_channels = np.random.randint(0, 3, size=self.num_aps)
        self.ap_powers = np.random.randint(0, 3, size=self.num_aps)
        self.client_connections = np.full(self.num_clients, -1, dtype=int)
        
        return self._get_observation(), {}

    def step(self, action):
        # 1. Apply Actions
        action = action.reshape(self.num_aps, 2)
        self.ap_channels = action[:, 0]
        self.ap_powers = action[:, 1]
        
        # 2. Get Snapshot
        if self.current_step >= len(self.timeslots):
             current_timeslot_data = pd.DataFrame(columns=self.df.columns)
        else:
            ts = self.timeslots[self.current_step]
            current_timeslot_data = self.df[self.df['timeslot'] == ts]
            
        # 3. Physics & Helper Logic
        self._update_client_connections(current_timeslot_data)
        total_rate, client_rates = self._calculate_network_rate(current_timeslot_data)
        
        # 4. Next Step
        self.current_step += 1
        terminated = self.current_step >= len(self.timeslots)
        truncated = False
        
        observation = self._get_observation()
        info = {"mean_rate": np.mean(client_rates) if len(client_rates)>0 else 0}
        
        return observation, total_rate, terminated, truncated, info

    def _update_client_connections(self, df_snapshot):
        # Sticky logic: same as before
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
        # Rate calc logic: same as before
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
            for other_a_idx in range(self.num_aps):
                if other_a_idx == a_idx: continue
                if self.ap_channels[other_a_idx] != my_channel: continue
                if (c_idx, other_a_idx) in signal_map:
                    i_dbm = signal_map[(c_idx, other_a_idx)]
                    interference_mw += 10**(i_dbm/10)
            
            sinr = s_mw / (noise_mu_w + interference_mw)
            rate = 20e6 * np.log2(1 + sinr)
            if clients_per_ap[a_idx] > 1: rate /= clients_per_ap[a_idx]
            rate_mbps = rate / 1e6
            total_rate += rate_mbps
            client_rates.append(rate_mbps)
            
        return total_rate, client_rates

    def _get_observation(self):
        # ---------------------------------------------------------
        # Refactored Feature Construction
        # ---------------------------------------------------------
        
        # 1. Edge List
        if self.current_step >= len(self.timeslots):
             current_timeslot_data = pd.DataFrame(columns=self.df.columns)
        else:
            ts = self.timeslots[self.current_step]
            current_timeslot_data = self.df[self.df['timeslot'] == ts]

        edge_index = []
        edge_attr = []
        
        # Normalize RSSI for edge weights (using reference style normalization if possible, or just positive)
        # Reference used transform_matrix/1e-3. Here we use RSSI+100.
        
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

        # 2. Node Features
        # AP: [1.0, ID_Norm, Ch_Norm, Pwr_Norm]
        # Client: [-1.0, ID_Norm, -1.0, -1.0]
        
        x = []
        # AP Nodes
        for i in range(self.num_aps):
            id_norm = i / max(1, self.num_aps - 1)
            ch_norm = self.ap_channels[i] / 2.0
            p_norm = self.ap_powers[i] / 2.0
            x.append([1.0, id_norm, ch_norm, p_norm])
            
        # Client Nodes
        for i in range(self.num_clients):
            id_norm = i / max(1, self.num_clients - 1)
            x.append([-1.0, id_norm, -1.0, -1.0])
            
        x = torch.tensor(x, dtype=torch.float)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
