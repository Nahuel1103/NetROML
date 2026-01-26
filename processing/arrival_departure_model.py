"""
Modelo de arribos y partidas para simular clientes que se conectan y desconectan
usando procesos estocásticos (Poisson para arribos, Exponencial para duración).
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ClientEvent:
    """Representa un evento de conexión de un cliente WiFi"""
    client_id: str
    arrival_time: int  # timestep de arribo (conexión)
    departure_time: int  # timestep de partida (desconexión)
    duration: int  # duración de la sesión en timesteps
    block_index: int  # índice del bloque de datos RSSI a utilizar (base 1)
    

class ArrivalDepartureModel:
    """
    Modelo estocástico de gestión de usuarios WiFi:
    - Arribos: Proceso de Poisson (N usuarios nuevos por timestep)
    - Duración: Distribución Exponencial (tiempo de permanencia)
    """
    
    def __init__(
        self,
        arrival_rate: float = 2.0,  # λ (lambda) Poisson: promedio de nuevos usuarios por timestep
        mean_duration: float = 10.0,  # μ (mu) Exponencial: duración promedio de sesión en timesteps
        total_timesteps: int = 100,
        random_seed: int = 314,
        client_block_counts: Dict[str, int] = None  # Mapa {mac_cliente: max_block_index} para datos reales
    ):
        """
        Inicializa el modelo de simulación.

        Args:
            arrival_rate: Tasa media de arribos por timestep (λ)
            mean_duration: Duración media de conexión (1/λ de la exponencial)
            total_timesteps: Horizonte temporal total de la simulación
            random_seed: Semilla para reproducibilidad de los eventos aleatorios
            client_block_counts: Diccionario opcional para asignar bloques de datos reales
                                a los clientes simulados.
        """
        self.arrival_rate = arrival_rate
        self.mean_duration = mean_duration
        self.total_timesteps = total_timesteps
        self.client_block_counts = client_block_counts or {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.client_counter = 0
        self.events: List[ClientEvent] = []
        
    def generate_arrivals(self, timestep: int) -> int:
        """
        Calcula cuántos clientes nuevos llegan en el timestep actual.
        Usa una distribución de Poisson.
        
        Args:
            timestep: El instante de tiempo actual
            
        Returns:
            int: Cantidad de nuevos clientes que se conectan
        """
        return np.random.poisson(self.arrival_rate)
    
    def generate_duration(self) -> int:
        """
        Determina cuánto tiempo permanecerá conectado un cliente.
        Usa una distribución Exponencial.
        
        Returns:
            int: Duración de la sesión en timesteps (mínimo 1)
        """
        # Muestreo exponencial
        duration = np.random.exponential(self.mean_duration)
        # Garantizar al menos 1 timestep de duración
        return max(1, int(np.round(duration)))
    
    def generate_block_index(self, mac_cliente: str) -> int:
        """
        Asigna aleatoriamente un bloque de datos histórico válido para un cliente.
        Esto permite variar los datos de RSSI usados en cada sesión simulada.
        
        Args:
            mac_cliente: Dirección MAC del cliente real
            
        Returns:
            int: Índice de bloque seleccionado (entre 1 y max_block disponible)
        """
        if mac_cliente in self.client_block_counts:
            max_block = self.client_block_counts[mac_cliente]
            # Selección uniforme de un bloque válido
            return np.random.randint(1, max_block + 1)
        else:
            # Valor por defecto si no hay datos históricos
            return 1
    
    def simulate_all_events(self) -> List[ClientEvent]:
        """
        Ejecuta la simulación completa pre-calculando todos los eventos de conexión.
        
        Si hay datos de clientes reales (client_block_counts), asigna identidades
        MAC reales y bloques de datos específicos a cada evento simulado.
        
        Returns:
            List[ClientEvent]: Cronograma completo de conexiones ordenado por tiempo
        """
        self.events = []
        self.client_counter = 0
        
        # Optimización: convertir claves a lista una sola vez
        available_clients = list(self.client_block_counts.keys()) if self.client_block_counts else []
        
        for t in range(self.total_timesteps):
            # 1. Determinar número de arribos (Poisson)
            num_arrivals = self.generate_arrivals(t)
            
            # 2. Generar eventos para cada nuevo cliente
            for _ in range(num_arrivals):
                duration = self.generate_duration()
                departure_time = t + duration
                
                # 3. Asignar identidad y datos
                if available_clients:
                    # Elegir un cliente real al azar
                    mac_cliente = np.random.choice(available_clients)
                    # Elegir un bloque de datos válido para ese cliente
                    block_idx = self.generate_block_index(mac_cliente)
                    # La ID del evento será la MAC real
                    client_id = mac_cliente
                else:
                    # Modo sintético (sin datos reales)
                    client_id = f"client_{self.client_counter}"
                    block_idx = 1
                
                event = ClientEvent(
                    client_id=client_id,
                    arrival_time=t,
                    departure_time=departure_time,
                    duration=duration,
                    block_index=block_idx
                )
                
                self.events.append(event)
                self.client_counter += 1
        
        return self.events
    
    def get_active_clients(self, timestep: int) -> List[ClientEvent]:
        """
        Identifica qué clientes están conectados en un instante específico.
        
        Args:
            timestep: El instante a consultar
            
        Returns:
            List[ClientEvent]: Lista de eventos activos en ese momento
        """
        return [
            event for event in self.events
            if event.arrival_time <= timestep < event.departure_time
        ]
    
    def get_arrivals_at_timestep(self, timestep: int) -> List[ClientEvent]:
        """
        Recupera los clientes que inician su conexión en el instante dado.
        """
        return [
            event for event in self.events
            if event.arrival_time == timestep
        ]
    
    def get_departures_at_timestep(self, timestep: int) -> List[ClientEvent]:
        """
        Recupera los clientes que finalizan su conexión en el instante dado.
        """
        return [
            event for event in self.events
            if event.departure_time == timestep
        ]
    
    def get_statistics(self) -> Dict:
        """
        Calcula métricas clave de la simulación realizada.
        
        Returns:
            Dict: Diccionario con estadísticas de duración, ocupación, etc.
        """
        if not self.events:
            return {}
        
        durations = [e.duration for e in self.events]
        
        # Calcular ocupación promedio por timestep
        occupancy_per_timestep = []
        for t in range(self.total_timesteps):
            active = len(self.get_active_clients(t))
            occupancy_per_timestep.append(active)
        
        return {
            'total_clients': len(self.events),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'mean_occupancy': np.mean(occupancy_per_timestep),
            'max_occupancy': np.max(occupancy_per_timestep),
            'total_timesteps': self.total_timesteps
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Crear modelo
    model = ArrivalDepartureModel(
        arrival_rate=3.0,  # promedio 3 clientes por timestep
        mean_duration=15.0,  # duración promedio 15 timesteps
        total_timesteps=50,
        random_seed=42
    )
    
    # Simular eventos
    events = model.simulate_all_events()
    
    print(f"Total de clientes generados: {len(events)}")
    print("\nPrimeros 10 eventos:")
    for event in events[:10]:
        print(f"  {event.client_id}: arriba en t={event.arrival_time}, parte en t={event.departure_time}, duración={event.duration}")
    
    # Ver clientes activos en timestep 10
    print(f"\nClientes activos en t=10:")
    active = model.get_active_clients(10)
    for event in active:
        print(f"  {event.client_id}")
    
    # Estadísticas
    print("\nEstadísticas:")
    stats = model.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
