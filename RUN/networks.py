import numpy as np
import scipy.io
import pdb
from scipy import sparse
import pickle


#############################################
############ Generate random ad-hoc (pair) network ###############
##############################################
# inputs:
# num_channels - number of users
# pl - pathloss factor
# outputs:
# L - matrix of pathloss fading

# Explained on Ad-hoc networks on Page 10 Alejandro and Mark's paper --Santiago
# def build_adhoc_network(num_channels,pl):

#     transmitters = np.random.uniform(low=-num_channels, high=num_channels, size=(num_channels,2))
#     receivers = transmitters + np.random.uniform(low=-num_channels/4,high=num_channels/4, size=(num_channels,2))

#     L = np.zeros((num_channels,num_channels))

#     for i in np.arange(num_channels):
#         for j in np.arange(num_channels):
#             d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
#             L[i,j] = np.power(d,-pl)


#     data = {'T': [], 'R':[], 'L':[]}
#     data['T'] = transmitters
#     data['R'] = receivers
#     data['A'] = L
#     #scipy.io.savemat("pl_net" + str(num_channels) + ".mat", data)
#     for i in range(0,3):
#         L[i,i] += 5
        
#     # L[0,2:6] = 0
#     L[1,3:6] = 0
#     L[0,3:6] = 0
#     # L[2,0] = 0
#     L[2,3:6] = 0
#     L[3,:] = 0
#     L[4,:] = 0
#     L[5,:] = 0

#     return L




# def build_adhoc_network(num_channels, pl, break_symmetry=True):

#     # Transmisores uniformemente en un área cuadrada
#     transmitters = np.random.uniform(low=-num_channels, high=num_channels, size=(num_channels,2))
#     receivers = transmitters + np.random.uniform(low=-num_channels/4, high=num_channels/4, size=(num_channels,2))

#     L = np.zeros((num_channels, num_channels))

#     for i in np.arange(num_channels):
#         for j in np.arange(num_channels):
#             d = np.linalg.norm(transmitters[i,:] - receivers[j,:])
#             L[i,j] = np.power(d, -pl)

#     data = {'T': [], 'R':[], 'L':[]}
#     data['T'] = transmitters
#     data['R'] = receivers
#     data['A'] = L
    
#     # MODIFICACIÓN: Diagonal asimétrica para romper simetría
# # Después de calcular L
#     if break_symmetry:
#         for i in range(0, 3):
#             L[i,i] += 5 + i
#             # Reducir interferencia cruzada entre activos
#             for j in range(0, 3):
#                 if i != j:
#                     L[i,j] *= 0.3  # Reducir a 30% la interferencia
#     else:
#         # Simétrico (original)
#         for i in range(0, 3):
#             L[i,i] += 5
    
#     # # Eliminar enlaces

#     # Escenario 1
#     # L[0,2:6] = 0
#     # L[1,3:6] = 0
#     # L[2,0] = 0
#     # L[2,3:6] = 0
#     # L[3,:] = 0
#     # L[4,:] = 0
#     # L[5,:] = 0

#     #Escenario 2
#     L[1,3:6] = 0
#     L[0,3:6] = 0
#     L[2,3:6] = 0
#     L[3,:] = 0
#     L[4,:] = 0
#     L[5,:] = 0
    
#     return L

def build_adhoc_network(pl):
    """
    Crea matriz 6x6 para 3 links (3 TX + 3 RX).
    Nodos 0,1,2 = TX (afuera)
    Nodos 3,4,5 = RX (adentro)
    """
    # 3 TX afuera (triángulo grande)
    tx_radius = 3.5
    transmitters = np.array([
        [0.0, tx_radius],                           
        [-tx_radius * np.sqrt(3)/2, -tx_radius/2], 
        [tx_radius * np.sqrt(3)/2, -tx_radius/2]   
    ])
    
    # 3 RX adentro (triángulo pequeño)
    receivers = np.array([
        [0.0, 0.3],           
        [-0.26, -0.15],       
        [0.26, -0.15]         
    ])
    
    # Combinar TODOS los nodos [TX0, TX1, TX2, RX0, RX1, RX2]
    all_nodes = np.vstack([transmitters, receivers])
    
    # Matriz 6x6: distancias entre TODOS los nodos
    L = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            d = np.linalg.norm(all_nodes[i] - all_nodes[j])
            d = max(d, 0.1)
            L[i,j] = np.power(d, -pl)

    L[1,3:6] = 0
    L[0,3:6] = 0
    L[2,3:6] = 0
    L[3,:] = 0
    L[4,:] = 0
    L[5,:] = 0
    
    return L
















#############################################
############ Generate line network ###############
##############################################
# inputs:
# num_channels - number of users
# pl - pathloss factor
# outputs:
# L - matrix of pathloss fading
def build_line_network(num_channels,pl):

    receivers = np.vstack([np.linspace(-num_channels/2,num_channels/2,num_channels), np.zeros(num_channels)]).transpose()

    transmitters = np.vstack([ receivers[:,0] + np.random.uniform(low=-1,high=1, size=(num_channels,)), np.random.uniform(low=-4,high=4, size=(num_channels,))]).transpose()

    L = np.zeros((num_channels,num_channels))

    for i in np.arange(num_channels):
        for j in np.arange(num_channels):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)


    data = {'T': [], 'R':[]}
    data['T'] = transmitters
    data['R'] = receivers
    scipy.io.savemat("line_net" + str(num_channels) + ".mat", data)
    return L

#############################################
############ Generate multi-cell network ###############
##############################################
# inputs:
# n - number of base stations
# k - number of cellular uses per base station
# pl - pathloss factor
# outputs:
# L - matrix of pathloss fading
# assign - kx1 vector assigning users to base station
def build_cellular_network(n, k, pl):
    M = n*k

    n_rows = np.floor(n/3)

    assign = np.kron(np.arange(n),np.ones((1,k))) # assignment of user to base station

    receivers = np.vstack([np.linspace(-M/2,M/2,n), np.zeros(n)]).transpose()
    boundaries = np.linspace(-3*M/4,3*M/4,n+1)

    transmitters_x = np.zeros((M,1))
    transmitters_y = np.zeros((M,1))
    for i in np.arange(n):
        transmitters_x[i*k:(i+1)*k] = receivers[i,0] + np.random.uniform(low=-10,high=10,size=(k,1))
        transmitters_y[i*k:(i+1)*k] = receivers[i,1] + np.random.uniform(low=-10,high=10,size=(k,1))

    transmitters = np.hstack([transmitters_x, transmitters_y])

    data = {'T': [], 'R':[], 'A':[]}
    data['T'] = transmitters
    data['R'] = receivers
    data['A'] = assign[0].astype(int)
    scipy.io.savemat("cell_net.mat", data)

    L = np.zeros((M,n))

    for i in np.arange(M):
        for j in np.arange(n):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)

    return L, assign[0].astype(int)



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


def build_adhoc_network_3d_choclo(num_channels, pl, break_symmetry=True):
    """
    Crea una red ad-hoc donde transmisores y receptores están distribuidos en 3D.
    La matriz L contiene los valores de enlace calculados según la distancia euclídea en 3D y el path loss pl.
    Si break_symmetry=True, se modifica la diagonal para romper simetría y reducir interferencia cruzada en los primeros 3 nodos.
    NO se ponen ceros de forma manual: la red queda totalmente conectada según la distancia.
    """
    # Transmisores uniformemente en un cubo
    transmitters = np.random.uniform(low=-num_channels, high=num_channels, size=(num_channels, 3))
    receivers = transmitters + np.random.uniform(low=-num_channels/4, high=num_channels/4, size=(num_channels, 3))

    L = np.zeros((num_channels, num_channels))

    # Calculo de la matriz de enlaces en 3D
    for i in np.arange(num_channels):
        for j in np.arange(num_channels):
            d = np.linalg.norm(transmitters[i,:] - receivers[j,:])
            # Si la distancia es cero, evitamos división por cero (self-loop)
            if d == 0:
                L[i,j] = 1.0
            else:
                L[i,j] = np.power(d, -pl)

    data = {'T': transmitters, 'R': receivers, 'A': L}

    if break_symmetry:
        for i in range(0, min(num_channels, 3)):
            L[i,i] += 5
            # Reducir interferencia cruzada entre los primeros 3
            for j in range(0, min(num_channels, 3)):
                if i != j:
                    L[i,j] *= 0.3
    else:
        # Simetría en diagonal (versión original)
        for i in range(0, min(num_channels, 3)):
            L[i,i] += 5

    return L


def build_adhoc_network_3d(pl, tx_z=0.0, rx_z=0.0):
    """
    Crea matriz 3x3 para 3 transmisores en 3D (triángulo grande) y 3 receptores superpuestos en el centro del triángulo.
    Nodos 0,1,2 = TX (triángulo grande, plano z=tx_z)
    Nodos 0,1,2 destino = RX (los tres en el punto central, plano z=rx_z)
    """
    # 3 TX: triángulo grande
    tx_radius = 3.5
    transmitters = np.array([
        [0.0, tx_radius],                           
        [-tx_radius * np.sqrt(3)/2, -tx_radius/2], 
        [tx_radius * np.sqrt(3)/2, -tx_radius/2]   
    ])
    

        # 3 RX adentro (triángulo pequeño)
    receivers = np.array([
        [0.0, 0.3],           
        [-0.26, -0.15],       
        [0.26, -0.15]         
    ])
    
    L = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            d = np.linalg.norm(transmitters[i,:] - receivers[j,:])
            L[i,j] = np.power(d, -pl)
    return L



def build_adhoc_network(num_channels,pl):

    transmitters = np.random.uniform(low=-num_channels, high=num_channels, size=(num_channels,2))
    receivers = transmitters + np.random.uniform(low=-num_channels/4,high=num_channels/4, size=(num_channels,2))

    L = np.zeros((num_channels,num_channels))

    for i in np.arange(num_channels):
        for j in np.arange(num_channels):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)


    data = {'T': [], 'R':[], 'L':[]}
    data['T'] = transmitters
    data['R'] = receivers
    data['A'] = L
    #scipy.io.savemat("pl_net" + str(num_channels) + ".mat", data)

    return L