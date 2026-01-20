import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv

class MultiHeadGNN(torch.nn.Module):
    """
    GNN con arquitectura Multi-Head para desacoplar la decisión de Canal 
    (incluyendo No-Tx) y la decisión de Nivel de Potencia.
    Incorpora Batch Normalization para estabilizar el entrenamiento.
    """
    def __init__(self, input_dim, hidden_dim, num_channels, num_power_levels, num_layers=5, dropout=0.0, K=3):
        super(MultiHeadGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.K = K

        # Listas de capas convolucionales y Batch Normalization
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # --- 1. Capa de Entrada ---
        self.convs.append(TAGConv(input_dim, hidden_dim, K=K))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # --- 2. Capas Intermedias (num_layers - 2) ---
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # --- 3. Última capa convolucional (Produce el embedding final) ---
        # Esta capa se incluye en la lista, pero su salida NO se normaliza ni activa.
        self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K)) 
        
        
        # --- HEADS DE PROYECCIÓN (Se aplican sobre el embedding final) ---
        
        # 1. Head de Canal: Predice 1 (No_Tx) + num_channels
        self.channel_head = nn.Linear(hidden_dim, num_channels + 1)
        
        # 2. Head de Potencia: Predice num_power_levels
        self.power_head = nn.Linear(hidden_dim, num_power_levels) 
        
        self.initialize_weights()

    def initialize_weights(self):
        """Inicialización Xavier (Glorot) para mejor flujo de gradientes."""
        for m in self.modules():
            if isinstance(m, TAGConv):
                # Inicialización para el peso lineal dentro de TAGConv
                if hasattr(m, 'lin'):
                    nn.init.xavier_uniform_(m.lin.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        
        # Pre-procesamiento de edge_attr para usarlo como edge_weight
        edge_weight = edge_attr.squeeze() if edge_attr is not None and edge_attr.dim() > 1 else edge_attr

        # Bucle de capas ocultas (num_layers - 1)
        # Convolución -> BatchNorm -> LeakyReLU -> Dropout
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x) 
            x = F.leaky_relu(x, 0.2)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Última capa convolucional (Produce el embedding x_embedding)
        # Esta es la capa self.convs[-1]
        x_embedding = self.convs[-1](x, edge_index, edge_weight)
        
        # --- Proyección de los Heads (Logits) ---
        channel_logits = self.channel_head(x_embedding)
        power_logits = self.power_head(x_embedding)

        # Retorna ambos sets de logits
        return channel_logits, power_logits