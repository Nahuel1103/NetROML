import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def join_graphs(train_list, val_list):
    """ Funcion que toma dos lista de archivso .pkl y devuelve dos listas con los datos unidos correspondientemente.

    Parameters: 
        train list (list): lista de archivos .pkl que contienen listas de grafos para entrenar.
        val list (list): lista de archivos .pkl que contienen listas de grafos para validar.

        
    Returns: 
        train_join_graphs (list): lista donde cada elemento es un grafo correspondiente a un instante de tiempo diferente
        val_join_graphs (list): lista donde cada elemento es un grafo correspondiente a un instante de tiempo diferente
    """

    train_join_graphs = []
    val_join_graphs = []
    
    for file in train_list:
        # cargo la lista de train
        with open(file, 'rb') as archivo:
            train_graphs = pickle.load(archivo)
        train_join_graphs.extend(train_graphs)
    
    if (len(val_list) > 0):
        for file in val_list:
            # cargo la lista de train
            with open(file, 'rb') as archivo:
                val_graphs = pickle.load(archivo)
            val_join_graphs.extend(val_graphs)

    return train_join_graphs, val_join_graphs

if __name__ == '__main__':
    
    banda = ['2_4', '5']
    buildings = [[32, False]]  
    for building_id, b5g in buildings:

        #val_months = ['marzo', 'abril', 'mayo', 'junio', 'julio']
        #train_months = ['agosto', 'setiembre', 'octubre', 'noviembre', 'diciembre']        

        train_months = ['marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto']
        val_months = ['setiembre', 'octubre', 'noviembre', 'diciembre']

        folder_name = str(banda[b5g]) + '_' + str(building_id) + '/'
        
        train_list = ['list_' + str(banda[b5g]) + '_graphs_' + str(building_id) + '_' + month + '.pkl' for month in train_months]
        train_list = [folder_name + file for file in train_list]

        val_list = ['list_' + str(banda[b5g]) + '_graphs_' + str(building_id) + '_' + month + '.pkl' for month in val_months]
        val_list = [folder_name + file for file in val_list]

        train_join_graphs, val_join_graphs = join_graphs(train_list=train_list, val_list=val_list)
        
        train_file = 'train_' + str(banda[b5g]) + '_graphs_' + str(building_id) + '.pkl'
        train_file = folder_name + train_file 
        # guardar la lista train_join_graphs en un archivo
        with open(train_file, 'wb') as archivo:
            pickle.dump(train_join_graphs, archivo)

        if (len(val_list) > 0): 
            val_file = 'val_' + str(banda[b5g]) + '_graphs_' + str(building_id) + '.pkl'
            val_file = folder_name + val_file 
            # guardar la lista val_final_file en un archivo
            with open(val_file, 'wb') as archivo:
                pickle.dump(val_join_graphs, archivo)
