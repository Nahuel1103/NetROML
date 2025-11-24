import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from fcnn import (FCNN,
                 compute_rate,
                 get_best_channel,
                 build_dataset)
from utils import graphs_to_tensor_sc

def train_fcnn(channel_matrix_tensor, batch_size=64, epochs=30, lr=1e-3):

    print(f"[INFO] Generando dataset supervisado a partir de {len(channel_matrix_tensor)} matrices...")

    X, Y = build_dataset(channel_matrix_tensor)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[INFO] Entrenando FCNN ({epochs} epochs)...\n")

    for epoch in range(epochs):
        total_loss = 0

        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss = {total_loss:.4f}")

    print("\n[INFO] Entrenamiento completado.")
    return model



# ======================================================================
# 6. EJEMPLO REAL — USANDO TU PIPELINE EXACTO
# ======================================================================
if __name__ == "__main__":

    print("\n==============================================")
    print("  CARGANDO MATRICES DE CANAL USANDO TU PIPELINE")
    print("==============================================\n")

    # 🔥 USANDO TUS DATOS SYNTHETIC DE NetROML
    x_tensor, channel_matrix_tensor = graphs_to_tensor_sc(
        num_links=3,
        num_features=1,
        b5g=False,
        building_id=990
    )

    # Entrenar FCNN
    model = train_fcnn(channel_matrix_tensor, epochs=35)

    # ---------------------------------------------
    # TEST: Predecir canal para la primera matriz
    # ---------------------------------------------
    test_H = channel_matrix_tensor[0].reshape(1, 9)
    logits = model(test_H)
    pred = torch.argmax(logits).item()

    print("\n==============================================")
    print(" MATRIZ DE PRUEBA:\n", channel_matrix_tensor[0])
    print("\n CANAL PREDICHO POR LA FCNN:", pred)
    print("==============================================\n")

