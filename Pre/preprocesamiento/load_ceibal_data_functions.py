import random
import argparse
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_build_ceibal_adhoc_network(b5g = True, building_id = 1190):
    banda = ['2_4', '5']
    # cargo la lista desde el archivo
    with open('list_' + str(banda[b5g]) + '_graphs_' + str(building_id) + '.pkl', 'rb') as archivo:
        graphs = pickle.load(archivo)
    return graphs

def transform_matrix(adj_matrix, all = True):

    if all:

        # primero defino quien es la pareja de quien.
        # recorro fila por fila la matriz y guardo en una lista el valor con mayor h y su posicion en la fila
        nodos = adj_matrix.shape[0]
        lista_parejas = []
        for i in range(nodos):
            # guardo el indice del nodo con menor h con el nodo i
            receptor = np.argmax(adj_matrix[i,:])
            # guardo el valor
            valor_maximo = adj_matrix[i,receptor]
            transmisor = i
            # guardo en una tupla el valor de h, el indice del transmisor y el indice del receptor
            elemento = [valor_maximo, transmisor, receptor]
            lista_parejas.append(elemento)

        H = np.zeros_like(adj_matrix)

        # voy rellenando la matriz
        for i in range(nodos):
            for j in range(nodos):
                if (i == j):
                    # en la diagonal uso los valores de h de lista_parejas con transmisor igual a i
                    H[i,j] = lista_parejas[i][0]
                else:
                    # en la no diagonal busco quien es el receptor del nodo j y relleno la matriz H con el h entre el nodo i y el receptor hallado
                    k = 0
                    nodo_receptor = 0
                    for k in range(nodos):
                        # busco la pareja que cumple que el nodo j es transmisor
                        if (lista_parejas[k][1] == j):
                            # guardo el nodo que recibe de j
                            nodo_receptor = lista_parejas[k][2]
                            if (nodo_receptor == i):
                                H[i,j] = 0.005
                            else:
                                H[i,j] = adj_matrix[i,nodo_receptor]
        return H
    
    else:
    
        # primero defino quien es transmisor y quien es receptor
        # defino los nodos
        nodos = list(np.arange(20))
        # barajo la lista
        random.seed(42)
        random.shuffle(nodos)
        # divido la lista barajada en dos listas de 10 elementos cada una
        nodos_tx = nodos[:10]
        nodos_rx = nodos[10:]

        # primero defino quien es la pareja de quien.
        # recorro fila por fila la matriz y guardo en una lista el valor con mayor h y su posicion en la fila
        nodos = adj_matrix.shape[0]
        lista_parejas = []
        for nodo_tx in nodos_tx:
            h_canal = []
            for nodo_rx in nodos_rx:
                h_canal.append(adj_matrix[nodo_tx,nodo_rx])
            index_pareja = np.argmax(h_canal)
            valor_maximo = adj_matrix[nodo_tx, nodos_rx[index_pareja]]
            # guardo en una tupla el valor de h, el indice del transmisor y el indice del receptor
            elemento = [valor_maximo, nodo_tx, nodos_rx[index_pareja]]
            lista_parejas.append(elemento)

        # defino matriz H que se usa en el articulo
        H = np.zeros((10,10))

        for i in np.arange(10):
            nodo_tx = lista_parejas[i][1]
            for j in np.arange(10):
                if (i == j):
                    H[i,j] = lista_parejas[i][0]
                else:
                    # si no estas en la diagonal, necesito el h entre el transmisor i y el receptor del transmisor j
                    nodo_tx_a_j = nodos_tx[j]
                    receptor_j = -1
                    # busco el nodo receptor del nodo transmisor j
                    for k in np.arange(10):
                        if lista_parejas[k][1] == nodo_tx_a_j:
                            receptor_j = k
                    H[i,j] = adj_matrix[nodos_tx[i], nodos_rx[receptor_j]]
        return H

