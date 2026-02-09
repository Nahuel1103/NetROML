import numpy as np

def build_adhoc_network_sc(num_links, pl=None):
    """
    Versión determinista de build_adhoc_network para sanity check.
    Mantiene la propiedad clave: RXs cerca de sus TXs.
    """
    
    if pl is None:
        pl = 2.5
    
    if num_links == 3:
        # Posiciones FIJAS (no aleatorias) pero con la misma lógica
        # TXs dispersos, RXs cerca de su TX
        
        transmitters = np.array([
            [0.0, 0.0],      # TX0 en origen
            [5.0, 0.0],      # TX1 a la derecha
            [2.5, 4.33]      # TX2 arriba (triángulo equilátero)
        ])
        
        # RXs: cada uno a distancia ~0.5 de su TX (fijo)
        receivers = np.array([
            [0.5, 0.0],      # RX0 cerca de TX0
            [5.5, 0.0],      # RX1 cerca de TX1
            [3.0, 4.33]      # RX2 cerca de TX2
        ])
        
    elif num_links == 5:
        # Pentágono regular
        radius = 5.0
        offset = 0.5  # Distancia TX-RX
        
        transmitters = np.zeros((5, 2))
        receivers = np.zeros((5, 2))
        
        for i in range(5):
            angle = 2 * np.pi * i / 5
            transmitters[i] = [radius * np.cos(angle), radius * np.sin(angle)]
            receivers[i] = [
                (radius + offset) * np.cos(angle),
                (radius + offset) * np.sin(angle)
            ]
    
    else:
        raise ValueError(f"num_links={num_links} no implementado")
    
    # Calcular matriz de canal
    L = np.zeros((num_links, num_links))
    
    for i in range(num_links):
        for j in range(num_links):
            d = np.linalg.norm(transmitters[i, :] - receivers[j, :])
            d = max(d, 0.1)  # Evitar división por cero
            L[i, j] = np.power(d, -pl)
    
    return L

def make_infinite_H_generator(num_links):
    """
    Crea un generador que produce infinitamente la misma matriz H
    generada por build_adhoc_network_sc.
    """
    # 1. Generamos H una sola vez
    H_fixed = build_adhoc_network_sc(num_links)
    print(f"H generada (se repetirá infinitamente):\n{H_fixed}\n")
    
    # 2. Loop infinito devolviendo esa misma referencia o una copia
    while True:
        # Devolvemos una copia para evitar que si se modifica externamente afecte al original
        yield H_fixed.copy()

if __name__ == "__main__":
    # Ejemplo de uso
    NUM_LINKS = 3
    
    # Instanciamos el generador
    h_iterator = make_infinite_H_generator(NUM_LINKS)
    
    print("--- Probando el iterador ---")
    for i in range(5):
        H_val = next(h_iterator)
        print(f"Iteración {i+1}: H[0,0] = {H_val[0,0]:.4f}")
    
    print("\nListo. Puedes usar 'h_iterator' en lugar de tu generador aleatorio habitual.")
