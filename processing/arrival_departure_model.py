"""
Modelo de arribos y partidas para simular clientes que se conectan y desconectan
usando procesos estocásticos (Poisson para arribos, Exponencial para duración).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ClientEvent:
    """Representa un evento de conexión de un cliente"""
    client_id: str
    arrival_time: int  # timestep de arribo (conexión)
    departure_time: int  # timestep de partida (desconexión)
    duration: int  # duración de la sesión en timesteps
    

# =========================
# Modelo Arribos-Partidas
# =========================

class ArrivalDepartureModel:
    """
    Modelo estocástico de gestión de usuarios WiFi:
    - Arribos: Proceso de Poisson
    - Duración: Distribución Exponencial
    """

    def __init__(
        self,
        arrival_rate: float = 2.0,
        mean_duration: float = 10.0,
        total_timesteps: int = 100,
        random_seed: int = 314,
        client_block_counts: Dict[str, int] = None
    ):
        self.arrival_rate = arrival_rate
        self.mean_duration = mean_duration
        self.total_timesteps = total_timesteps
        self.client_block_counts = client_block_counts or {}

        if random_seed is not None:
            np.random.seed(random_seed)

        self.client_counter = 0
        self.events: List[ClientEvent] = []

        # Cache para clientes reales
        self.available_clients = list(self.client_block_counts.keys())

    # -------------------------
    # Generadores estocásticos
    # -------------------------

    def generate_arrivals(self) -> int:
        """Número de arribos en un timestep (Poisson)."""
        return np.random.poisson(self.arrival_rate)

    def generate_duration(self) -> int:
        """Duración de sesión (Exponencial, mínimo 1)."""
        duration = np.random.exponential(self.mean_duration)
        return max(1, int(np.round(duration)))

    # -------------------------
    # Simulación
    # -------------------------

    def simulate_all_events(self) -> List[ClientEvent]:
        """Precalcula todos los eventos de conexión."""
        self.events = []
        self.client_counter = 0

        for t in range(self.total_timesteps):
            num_arrivals = self.generate_arrivals()

            for _ in range(num_arrivals):
                duration = self.generate_duration()
                departure_time = t + duration

                if self.available_clients:
                    client_id = np.random.choice(self.available_clients)
                else:
                    client_id = f"client_{self.client_counter}"

                event = ClientEvent(
                    client_id=client_id,
                    arrival_time=t,
                    departure_time=departure_time,
                    duration=duration
                )

                self.events.append(event)
                self.client_counter += 1

        return self.events

    # -------------------------
    # Queries temporales
    # -------------------------

    def get_active_clients(self, timestep: int) -> List[ClientEvent]:
        """Clientes activos en un timestep."""
        return [e for e in self.events if e.arrival_time <= timestep < e.departure_time]

    def get_arrivals_at_timestep(self, timestep: int) -> List[ClientEvent]:
        return [e for e in self.events if e.arrival_time == timestep]

    def get_departures_at_timestep(self, timestep: int) -> List[ClientEvent]:
        return [e for e in self.events if e.departure_time == timestep]

    # -------------------------
    # Sampling de bloques RSSI
    # -------------------------

    def get_block_index_at_timestep(self, event: ClientEvent, timestep: int) -> int:
        """
        Sortea con reposición el bloque RSSI a usar por el cliente
        en un timestep específico.
        """
        if not (event.arrival_time <= timestep < event.departure_time):
            raise ValueError("Cliente no activo en este timestep")

        mac = event.client_id

        if mac in self.client_block_counts:
            max_block = self.client_block_counts[mac]
            return np.random.randint(1, max_block + 1)
        else:
            return 1

    # -------------------------
    # Estadísticas
    # -------------------------

    def get_statistics(self) -> Dict:
        if not self.events:
            return {}

        durations = [e.duration for e in self.events]
        occupancy = [len(self.get_active_clients(t)) for t in range(self.total_timesteps)]

        return {
            "total_clients": len(self.events),
            "mean_duration": float(np.mean(durations)),
            "std_duration": float(np.std(durations)),
            "min_duration": int(np.min(durations)),
            "max_duration": int(np.max(durations)),
            "mean_occupancy": float(np.mean(occupancy)),
            "max_occupancy": int(np.max(occupancy)),
            "total_timesteps": self.total_timesteps,
        }

def build_client_block_counts(df: pd.DataFrame) -> dict:
    """
    Construye el diccionario {mac_cliente: max_block_index}
    a partir del DataFrame de RSSI.
    """
    return (
        df.groupby("mac_cliente")["block_index"]
        .max()
        .astype(int)
        .to_dict()
    )
