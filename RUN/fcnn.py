import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------
# 1) Modelo: Fully Connected Neural Network
# -------------------------------------------------------

class FCNN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def compute_rate(H, channel):
    """
    H: tensor (3,3) matriz de canal
    channel: entero {0,1,2}
    """

    # Asumimos:
    #   - El link usa el canal elegido
    #   - Interferencia = suma del resto de la matriz
    #   - señal = diagonal correspondiente al canal

    H = H.float()

    signal = H[channel, channel]
    interference = H[channel].sum() - signal
    noise = 1e-3   # ruido base

    sinr = signal / (interference + noise)
    rate = torch.log1p(sinr)

    return rate.item()

# ======================================================
# 3. TARGET: MEJOR DECISIÓN DE CANAL (CLASE 0/1/2)
# ======================================================
def get_best_channel(H):
    best_rate = -1e9
    best_ch = 0

    for ch in range(3):
        rate = compute_rate(H, ch)
        if rate > best_rate:
            best_rate = rate
            best_ch = ch

    return best_ch


# ======================================================
# 4. PREPROCESO: FLATTEN MATRICES Y ARMAR DATASET
# ======================================================
def build_dataset(channel_matrices):
    """
    channel_matrices: numpy array (N, 3,3)
    """
    X = []
    y = []

    for H in channel_matrices:
        H_tensor = torch.tensor(H)

        # entrada (9 features)
        X.append(H_tensor.reshape(-1))

        # target: canal óptimo
        best_ch = get_best_channel(H_tensor)
        y.append(best_ch)

    X = torch.stack(X, dim=0)  # (N, 9)
    y = torch.tensor(y, dtype=torch.long)  # (N,)

    return X, y

