import pickle
import torch
import numpy as np
import os

# Crear grafos sintéticos simples para testing
def create_simple_graphs():
    # Grafo de 3 nodos - completamente conectado
    graph_3_nodes = np.array([
        [0, 1, 1],
        [1, 0, 1], 
        [1, 1, 0]
    ], dtype=np.float32)
    
    # Grafo de 4 nodos - estrella
    graph_4_nodes_star = np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ], dtype=np.float32)
    
    # Grafo de 4 nodos - cadena
    graph_4_nodes_chain = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    graphs = [graph_3_nodes, graph_4_nodes_star, graph_4_nodes_chain]
    
    # Guardar en la estructura de directorios esperada
    path = '/Users/nahuelpineyro/NetROML/graphs/2_4_990/'
    os.makedirs(path, exist_ok=True)
    
    with open(path + 'synthetic_graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)
    
    print(f"Grafos de test guardados en: {path}synthetic_graphs.pkl")
    print("Grafos creados:")
    for i, graph in enumerate(graphs):
        print(f"Grafo {i+1} ({graph.shape[0]} nodos):")
        print(graph)

if __name__ == '__main__':
    create_simple_graphs()
