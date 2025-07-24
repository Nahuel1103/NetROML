import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pickle


def plot_rates(val_rates_path, baseline1_rates_path, baseline2_rates_path):
    """
    Function that receives paths to different .pkl files and plot all the files together.
    Each file has objective function values corresponding to different systems configurations.  
    """

    val_pkl = val_rates_path + 'objective_function_values_val_150.pkl'
    with open(val_pkl, 'rb') as file:
        val_rates = pickle.load(file)

    baseline1_pkl = baseline1_rates_path + 'baseline1_150.pkl'
    with open(baseline1_pkl, 'rb') as file:
        baseline1_rates = pickle.load(file)

    
    baseline2_pkl = baseline2_rates_path + 'baseline2_150.pkl'
    with open(baseline2_pkl, 'rb') as file:
        baseline2_rates = pickle.load(file)

    plt.figure(figsize=(16,9))
    plt.title('Funcion Objetivo')
    plt.xlabel('Iteraciones (x10)')
    plt.ylabel('Capacidad')
    plt.plot(val_rates, label='validation')
    plt.plot(baseline1_rates, label='baseline 1')
    plt.plot(baseline2_rates, label='baseline 2')
    plt.grid()
    plt.legend()
    image_name = val_rates_path + 'all_objective_functions' + '.png'
    plt.savefig(image_name)
    plt.close()


if __name__ == '__main__':

    val_rates_path = '../results/2_4_856/torch_results/n_layers5_order3/ceibal_val_5e-04_5e-04_64_0_267309_502321/'
    baseline1_rates_path = '../results/2_4_856/torch_results/n_layers5_order3/ceibal_train_5e-05_1e-04_64_150_267309_502321_baseline1/'
    baseline2_rates_path = '../results/2_4_856/torch_results/n_layers5_order3/ceibal_train_5e-05_1e-04_64_150_267309_502321_baseline2/'

    plot_rates(val_rates_path=val_rates_path, baseline1_rates_path=baseline1_rates_path, baseline2_rates_path=baseline2_rates_path)