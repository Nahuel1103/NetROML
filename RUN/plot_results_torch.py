import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def plot_results(building_id, b5g, normalized_psi, normalized_psi_values=[], num_layers=5, K=3, batch_size=64, epochs = 100, rn=100, rn1=100, eps=5e-5, mu_lr=1e-4, objective_function_values=[], power_constraint_values=[], loss_values=[],
                 mu_k_values=[], baseline=0, mark=0, train=True):
    """
    Function that receives the values for different parameters resulting from some training
    of the system as lists and plots them. The corresponding plots are saved in the corresponding
    path, taking into consideration the building_id and the frequency band (2_4, 5) the network
    is using.
    """

    band = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str= str(f"{mu_lr:.0e}")
    batch_size_str = str(batch_size)

    if train:
        if mark:
            path = '/Users/nahuelpineyro/NetROML/results/' + str(band[b5g]) + '_' + str(building_id) + '/torch_results/n_layers' + str(num_layers) + '_order' + str(K) + '/mark_' + eps_str +  '_' + mu_lr_str + '_' + str(batch_size) + '_' + str(epochs) + '_' + str(rn) + '_' + str(rn1)
        else:
            path = '/Users/nahuelpineyro/NetROML/results/' + str(band[b5g]) + '_' + str(building_id) + '/torch_results/n_layers' + str(num_layers) + '_order' + str(K) + '/ceibal_train_' + eps_str +  '_' + mu_lr_str + '_' + str(batch_size) + '_' + str(epochs) + '_' + str(rn) + '_' + str(rn1)

        if (baseline==0):
            path = path + '/'
        else:
            path = path + '_baseline' + str(baseline) + '/'
    else:
        path = '/Users/nahuelpineyro/NetROML//results/' + str(band[b5g]) + '_' + str(building_id) + '/torch_results/n_layers' + str(num_layers) + '_order' + str(K) + '/ceibal_val_' + eps_str +  '_' + mu_lr_str + '_' + str(batch_size) + '_' + str(epochs) + '_' + str(rn) + '_' + str(rn1) + '/'


    if not os.path.exists(path):
        os.makedirs(path)

    if len(objective_function_values) > 0:

        plt.figure(figsize=(16,9))
        plt.title('Funcion Objetivo')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Capacidad')
        plt.plot(objective_function_values)
        plt.grid()
        image_name = 'objective_function'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path)
        plt.close()

        # plt.figure(figsize=(16,9))
        # plt.title('Funcion Objetivo post 2000')
        # plt.xlabel('Iteraciones (x10)')
        # plt.ylabel('Capacidad')
        # plt.plot(objective_function_values[2000:])
        # plt.grid()
        # image_name = f'objective_function_post2000'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        # image_path = os.path.join(path, image_name)
        # plt.savefig(image_path)
        # plt.close()

    if len(power_constraint_values) > 0:

        plt.figure(figsize=(16,9))
        plt.title('Restriccion de potencia')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Potencia')
        plt.plot(power_constraint_values)
        plt.grid()
        image_name = f'power constraint'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path)
        plt.close()

        # plt.figure(figsize=(16,9))
        # plt.title('Restriccion de potencia post 2000')
        # plt.xlabel('Iteraciones (x10)')
        # plt.ylabel('Potencia')
        # plt.grid()        
        # plt.plot(power_constraint_values[2000:])
        # image_name = f'power_constraint_post2000'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        # image_path = os.path.join(path, image_name)
        # plt.savefig(image_path)
        # plt.close()

    if len(loss_values) > 0:

        plt.figure(figsize=(16,9))
        plt.title('Loss')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Loss')
        plt.plot(loss_values)
        plt.grid()
        image_name = f'loss'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path)
        plt.close()

        plt.figure(figsize=(16,9))
        plt.title('Loss post 1000:1200')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Loss')
        plt.plot(loss_values[1000:1200])
        plt.grid()
        image_name = f'loss_bt1000:1200'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path)
        plt.close()

    if len(mu_k_values) > 0:

        plt.figure(figsize=(16,9))
        plt.title('mu_k')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('mu_k')
        plt.plot(mu_k_values)
        plt.grid()
        image_name = f'mu_k'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path)
        plt.close()

        # plt.figure(figsize=(16,9))
        # plt.title('mu_k post 2000')
        # plt.xlabel('Iteraciones (x10)')
        # plt.ylabel('mu_k')
        # plt.plot(mu_k_values[2000:])
        # plt.grid()
        # image_name = f'mu_k_post2000'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        # image_path = os.path.join(path, image_name)
        # plt.savefig(image_path)
        # plt.close()


        if (len(normalized_psi_values) > 0):
            # Generar nombres para las 7 políticas
            # nombres_politicas = [
            #     'Apagado (potencia 0)',
            #     'Canal 0 con potencia p0/2',
            #     'Canal 1 con potencia p0/2', 
            #     'Canal 2 con potencia p0/2',
            #     'Canal 0 con potencia p0',
            #     'Canal 1 con potencia p0',
            #     'Canal 2 con potencia p0'
            # ]

            normalized_psi_value = np.array(normalized_psi_values)
            plt.figure(figsize=(16, 9))
            
            # Graficar cada política (cada columna de la matriz)
            for i in range(normalized_psi_value.shape[1]):  # Iterar sobre las 7 columnas
                # plt.plot(normalized_psi_value[:, i], label=nombres_politicas[i])
                plt.plot(normalized_psi_value[:, i])

            
            plt.grid()
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel('Tiempo')
            plt.ylabel('Valor de Política Normalizado')
            plt.title('Políticas de Asignación de Potencia por Canal')
            
            image_name = 'policies.png'
            image_path = os.path.join(path, image_name)
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()

    normalized_psi= torch.mean(normalized_psi, dim=0)
    normalized_psi_array = normalized_psi.detach().numpy()
    psi_path = os.path.join(path, 'normalized_psi' + '_' + eps_str + '_' +  mu_lr_str + '_' + batch_size_str +'.txt')
    np.savetxt(psi_path, normalized_psi_array, delimiter=',', fmt='%.4f') 

    return path