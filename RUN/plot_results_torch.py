import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def plot_results(building_id, b5g, tx_probs, channel_probs, power_probs, 
                 tx_policy_values=[], channel_policy_values=[], power_policy_values=[],
                 num_layers=5, K=3, batch_size=64, epochs=100, rn=100, rn1=100, 
                 eps=5e-5, mu_lr=1e-4, objective_function_values=[], power_constraint_values=[], 
                 loss_values=[], mu_k_values=[], baseline=0, synthetic=0, train=True,
                 num_channels=3, num_power_levels=2):
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
        if synthetic:
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

        # Plot existing metrics (unchanged)
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
        plt.grid()
        image_name = f'power_constraint'+ '_' + eps_str + '_' + mu_lr_str + '_' + batch_size_str + '.png'
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

    # =================== NEW HIERARCHICAL POLICY PLOTS ===================
    
    # 1. Plot TX Decision Evolution
    if len(tx_policy_values) > 0:
        tx_policy_array = np.array(tx_policy_values)
        
        plt.figure(figsize=(16,9))
        plt.plot(tx_policy_array[:, 0], label='P(No Transmitir)', linewidth=2)
        plt.plot(tx_policy_array[:, 1], label='P(Transmitir)', linewidth=2)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Probabilidad')
        plt.title('Evolución de Política de Transmisión')
        
        image_name = 'tx_policy_evolution.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

    # 2. Plot Channel Selection Evolution
    if len(channel_policy_values) > 0:
        channel_policy_array = np.array(channel_policy_values)
        
        plt.figure(figsize=(16,9))
        for c in range(num_channels):
            plt.plot(channel_policy_array[:, c], 
                    label=f'P(Canal {c})', linewidth=2)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Probabilidad')
        plt.title('Evolución de Política de Selección de Canal')
        
        image_name = 'channel_policy_evolution.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

    # 3. Plot Power Level Evolution
    if len(power_policy_values) > 0:
        power_policy_array = np.array(power_policy_values)
        power_labels = ['P(p₀/2)', 'P(p₀)'] if num_power_levels == 2 else [f'P(Nivel {i})' for i in range(num_power_levels)]
        
        plt.figure(figsize=(16,9))
        for p in range(num_power_levels):
            plt.plot(power_policy_array[:, p], 
                    label=power_labels[p], linewidth=2)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel('Iteraciones (x10)')
        plt.ylabel('Probabilidad')
        plt.title('Evolución de Política de Nivel de Potencia')
        
        image_name = 'power_policy_evolution.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

    # 4. Combined Policy Plot (All decisions in subplots)
    if len(tx_policy_values) > 0 and len(channel_policy_values) > 0 and len(power_policy_values) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        
        # TX Policy
        ax1.plot(tx_policy_array[:, 0], label='No Tx', linewidth=2)
        ax1.plot(tx_policy_array[:, 1], label='Tx', linewidth=2)
        ax1.set_ylabel('Probabilidad TX')
        ax1.set_title('Políticas Jerárquicas - Evolución Completa')
        ax1.grid()
        ax1.legend()
        
        # Channel Policy
        for c in range(num_channels):
            ax2.plot(channel_policy_array[:, c], label=f'Canal {c}', linewidth=2)
        ax2.set_ylabel('Probabilidad Canal')
        ax2.grid()
        ax2.legend()
        
        # Power Policy
        for p in range(num_power_levels):
            ax3.plot(power_policy_array[:, p], label=power_labels[p], linewidth=2)
        ax3.set_ylabel('Probabilidad Potencia')
        ax3.set_xlabel('Iteraciones (x10)')
        ax3.grid()
        ax3.legend()
        
        plt.tight_layout()
        image_name = 'hierarchical_policies_combined.png'
        image_path = os.path.join(path, image_name)
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

    # 5. Final Policy State (Bar plots)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Final TX probabilities
    final_tx = torch.mean(tx_probs, dim=[0,1]).detach().numpy()
    ax1.bar(['No Tx', 'Tx'], final_tx, color=['red', 'green'], alpha=0.7)
    ax1.set_ylabel('Probabilidad')
    ax1.set_title('Estado Final: Decisión TX')
    ax1.grid(axis='y')
    
    # Final Channel probabilities
    final_channel = torch.mean(channel_probs, dim=[0,1]).detach().numpy()
    ax2.bar([f'Canal {i}' for i in range(num_channels)], final_channel, alpha=0.7)
    ax2.set_ylabel('Probabilidad')
    ax2.set_title('Estado Final: Selección de Canal')
    ax2.grid(axis='y')
    
    # Final Power probabilities
    final_power = torch.mean(power_probs, dim=[0,1]).detach().numpy()
    ax3.bar(power_labels, final_power, alpha=0.7)
    ax3.set_ylabel('Probabilidad')
    ax3.set_title('Estado Final: Nivel de Potencia')
    ax3.grid(axis='y')
    
    plt.tight_layout()
    image_name = 'final_policy_states.png'
    image_path = os.path.join(path, image_name)
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

    return path