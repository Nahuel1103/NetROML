import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def plot_results(building_id, b5g, normalized_psi, normalized_psi_values=[], power_values=[], 
                num_layers=5, K=3, batch_size=64, epochs = 100, rn=100, rn1=100, eps=5e-5, 
                mu_lr=1e-4, objective_function_values=[], power_constraint_values=[], 
                loss_values=[], mu_k_values=[], baseline=0, mark=0, train=True, num_channels = None,
            num_power_levels = None, power_levels = None):

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
        path = '/Users/nahuelpineyro/NetROML/results/' + str(band[b5g]) + '_' + str(building_id) + '/torch_results/n_layers' + str(num_layers) + '_order' + str(K) + '/ceibal_val_' + eps_str +  '_' + mu_lr_str + '_' + str(batch_size) + '_' + str(epochs) + '_' + str(rn) + '_' + str(rn1) + '/'


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



# Versión 1
    if (len(normalized_psi_values) > 0):
        normalized_psi_array = np.array(normalized_psi_values) 

        print(normalized_psi_array.shape)
        
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
            
            image_name = f'node_{node_idx}_channel_probs.svg'
            image_path = os.path.join(policy_dir, image_name)  # Guardar en subcarpeta
            plt.savefig(image_path)
            plt.close()
        
        final_probs = normalized_psi_array[-1]  # Última iteración           
    print(f"Probabilidades finales por nodo (solo canales):")
    for node_idx in range(num_nodes):
        # Solo canales de transmisión (ignorar el primer elemento "No TX")
        channel_probs = final_probs[node_idx, 1:] if len(final_probs.shape) > 1 else final_probs[1:]
        best_channel = np.argmax(channel_probs)
        best_prob = channel_probs[best_channel]
        print(f"  Nodo {node_idx}: Canal {best_channel} ({best_prob:.3f})")


# Versión 2
    # if (len(normalized_psi_values) > 0):
    #     normalized_psi_array = np.array(normalized_psi_values)  # [iterations, num_links, num_actions]
        
    #     num_nodes = normalized_psi_array.shape[1]
    #     num_actions = normalized_psi_array.shape[2]
        
    #     # Usar las variables conocidas
    #     num_channels = num_channels
    #     num_power_levels = num_power_levels
        
    #     # Verificar consistencia
    #     expected_actions = 1 + num_channels * num_power_levels
    #     if num_actions != expected_actions:
    #         print(f"Advertencia: Número de acciones ({num_actions}) no coincide con esperado ({expected_actions})")
        
    #     # Crear carpeta para políticas por nodo
    #     policy_dir = os.path.join(path, 'policy_per_node')
    #     if not os.path.exists(policy_dir):
    #         os.makedirs(policy_dir)
        
    #     # Plot para cada nodo: todas las acciones en un solo gráfico
    #     for node_idx in range(num_nodes):
    #         plt.figure(figsize=(14, 8))
            
    #         plt.title(f'Política de Transmisión - Nodo {node_idx}')
    #         plt.xlabel('Iteraciones (x10)')
    #         plt.ylabel('Probabilidad')
    #         plt.ylim(0, 1)
    #         plt.grid(True)
            
    #         # Plotear cada acción
    #         for action_idx in range(num_actions):
    #             if action_idx == 0:
    #                 # Acción "No TX"
    #                 label = 'No TX (0 mW)'
    #                 color = 'red'
    #                 linewidth = 3
    #                 linestyle = '-'
    #             else:
    #                 # Calcular canal y nivel de potencia para esta acción
    #                 action_offset = action_idx - 1
    #                 channel_idx = action_offset % num_channels
    #                 power_level_idx = action_offset // num_channels
                    
    #                 # Obtener el valor real de potencia en mW
    #                 power_value = power_levels[power_level_idx].item()  # Convertir a valor numérico
                    
    #                 label = f'Canal {channel_idx} - {power_value:.0f} mW'
    #                 color = f'C{channel_idx}'
    #                 linewidth = 2 - (power_level_idx * 0.3)
    #                 linestyle = ['-', '--', '-.', ':'][power_level_idx % 4]
            
    #             plt.plot(normalized_psi_array[:, node_idx, action_idx], 
    #                     label=label, color=color, linewidth=linewidth, 
    #                     linestyle=linestyle)
            
    #         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #         plt.tight_layout()
            
    #         image_name = f'node_{node_idx}_policy_analysis.png'
    #         image_path = os.path.join(policy_dir, image_name)
    #         plt.savefig(image_path, bbox_inches='tight')
    #         plt.close()



    normalized_psi= torch.mean(normalized_psi, dim=0)
    normalized_psi_array = normalized_psi.detach().numpy()
    psi_path = os.path.join(path, 'normalized_psi' + '_' + eps_str + '_' +  mu_lr_str + '_' + batch_size_str +'.txt')
    np.savetxt(psi_path, normalized_psi_array, delimiter=',', fmt='%.4f') 

    return path