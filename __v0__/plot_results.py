import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def plot_results(building_id, b5g, normalized_psi, normalized_psi_values=[], num_layers=5, K=3, batch_size=64, epochs = 100, rn=100, rn1=100, eps=5e-5, mu_lr=1e-4, objective_function_values=[], power_constraint_values=[], loss_values=[],
                 mu_k_values=[], baseline=0, mark=0, train=True, synthetic=0):

    """
    Function that plots training results with correct interpretation of probabilities per link.
    
    Args:
        normalized_psi_values: List of arrays with shape [num_links, num_actions]
                              where num_actions = num_channels + 1 (including "no TX")
    """

    band = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str = str(f"{mu_lr:.0e}")
    batch_size_str = str(batch_size)

    # Agrega esto:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_ROOT = os.path.join(BASE_DIR, '..', 'results')  # Ajusta si tu estructura cambia

    # Y reemplaza toda la lógica de creación de 'path' con esto:
    
    subfolder = f"{band[b5g]}_{building_id}/gymnasium_results/n_layers{num_layers}_order{K}_synthetic{synthetic}/__v0__"
    if train:
        if mark:
            exp_name = f"mark_{eps_str}_{mu_lr_str}_{batch_size}_{epochs}_{rn}_{rn1}"
        else:
            exp_name = f"ceibal_train_{eps_str}_{mu_lr_str}_{batch_size}_{epochs}_{rn}_{rn1}"
        if baseline == 0:
            result_path = os.path.join(RESULTS_ROOT, subfolder, exp_name)
        else:
            result_path = os.path.join(RESULTS_ROOT, subfolder, exp_name + f"_baseline{baseline}")
    else:
        exp_name = f"ceibal_val_{eps_str}_{mu_lr_str}_{batch_size}_{epochs}_{rn}_{rn1}"
        result_path = os.path.join(RESULTS_ROOT, subfolder, exp_name)
    path = result_path  # El resto de la función puede seguir usando 'path'

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

        plt.figure(figsize=(16,9))
        plt.title('Funcion Objetivo post 2000')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Capacidad')
        plt.plot(objective_function_values[2000:])
        plt.grid()
        image_name = f'objective_function_post2000'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path)
        plt.close()

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

        plt.figure(figsize=(16,9))
        plt.title('Restriccion de potencia post 2000')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Potencia')
        plt.grid()        
        plt.plot(power_constraint_values[2000:])
        image_name = f'power_constraint_post2000'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
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

        plt.figure(figsize=(16,9))
        plt.title('Loss post 2000')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Loss')
        plt.plot(loss_values[2000:])
        plt.grid()
        image_name = f'loss_post2000'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
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

        plt.figure(figsize=(16,9))
        plt.title('mu_k post 2000')
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('mu_k')
        plt.plot(mu_k_values[2000:])
        plt.grid()
        image_name = f'mu_k_post2000'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path)
        plt.close()
    
    if (len(normalized_psi_values) > 0):
        plt.figure(figsize=(16,9))
        plt.plot(normalized_psi_values)
        plt.grid()
        image_name = 'policies.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path)
        plt.close()

    # Handling tensor to numpy conversion if needed
    if isinstance(normalized_psi, torch.Tensor):
        normalized_psi = normalized_psi.detach().numpy()
        
    normalized_psi_mean = np.mean(normalized_psi, axis=0) 

    
    psi_path = os.path.join(path, 'normalized_psi' + '_' + eps_str + '_' +  mu_lr_str + '_' + batch_size_str +'.txt')

    return path
