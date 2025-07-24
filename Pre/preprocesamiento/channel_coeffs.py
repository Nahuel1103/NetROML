import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath('../wireless_learning_torch/'))
from utils import graphs_to_tensor, get_gnn_inputs


if __name__ == '__main__':
    
    band = ['2_4', '5']
    buildings = [[990, False, 5]]
    for building_id, b5g, num_channels in buildings:

        x_tensor, channel_matrix_tensor = graphs_to_tensor(train=False, num_channels=num_channels, num_features=1, b5g=b5g, building_id=building_id)
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        path = '../graphs/' + str(band[b5g]) + '_' + str(building_id) + '/' + 'channel_matrix_coefficients_' + str(building_id) + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        dim = dataset[0].matrix.shape[0]
        lista_entradas = [[] for _ in range(dim*dim)]

        for index, element in enumerate(dataset):
            H = element.matrix
            H = H/1e3
            flat_matrix = np.array(H).flatten()
            for idx, value in enumerate(flat_matrix):
                if value != 5e-3:
                    lista_entradas[idx].append(value)

        for index, entrada in enumerate(lista_entradas):
            np.savetxt(path + 'coeff_{}_values.txt'.format(index), entrada, delimiter=',') 
            plt.plot(entrada)
            plt.savefig(path + 'coeff_{}.png'.format(index))
            plt.close()