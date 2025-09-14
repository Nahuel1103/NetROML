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
            plt.ylim(-2, 1)  # ← AÑADIR ESTA LÍNEA
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
        normalized_psi_array = np.array(normalized_psi_values)  # [iterations, num_links, num_channels+1]
        
        num_nodes = normalized_psi_array.shape[1]
        num_channels = normalized_psi_array.shape[2]
        
        # Crear carpeta para políticas por nodo
        policy_dir = os.path.join(path, 'policy_per_node')
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
        
        # Plot para cada nodo: todos sus canales
        for node_idx in range(num_nodes):
            plt.figure(figsize=(16, 12))
            plt.title(f'Probabilidades de Canales - Nodo {node_idx}')
            plt.xlabel('Iteraciones (x10)')
            plt.ylabel('Probabilidad')
            
            # Plotear cada canal del nodo
            for channel_idx in range(num_channels):
                if channel_idx == 0:
                    label = f'No TX'
                else:
                    label = f'Canal {channel_idx-1}'
                
                plt.plot(normalized_psi_array[:, node_idx, channel_idx], 
                        label=label, linewidth=2)
            
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1)  # Probabilidades entre 0 y 1
            
            image_name = f'node_{node_idx}_channel_probs.png'
            image_path = os.path.join(policy_dir, image_name)  # Guardar en subcarpeta
            plt.savefig(image_path)
            plt.close()
        
        # También guardar CSV de políticas en la subcarpeta
        final_probs = normalized_psi_array[-1]  # Última iteración        
  
    
    # Estadísticas finales SIMPLIFICADAS (sin "No TX")
    print(f"Probabilidades finales por nodo (solo canales):")
    for node_idx in range(num_nodes):
        # Solo canales de transmisión (ignorar el primer elemento "No TX")
        channel_probs = final_probs[node_idx, 1:] if len(final_probs.shape) > 1 else final_probs[1:]
        best_channel = np.argmax(channel_probs)
        best_prob = channel_probs[best_channel]
        print(f"  Nodo {node_idx}: Canal {best_channel} ({best_prob:.3f})")

    normalized_psi= torch.mean(normalized_psi, dim=0)
    normalized_psi_array = normalized_psi.detach().numpy()
    psi_path = os.path.join(path, 'normalized_psi' + '_' + eps_str + '_' +  mu_lr_str + '_' + batch_size_str +'.txt')
    np.savetxt(psi_path, normalized_psi_array, delimiter=',', fmt='%.4f') 

    return path