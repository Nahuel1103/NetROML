import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def plot_results(building_id, b5g, normalized_psi, normalized_psi_values=[], power_values=[], 
                num_layers=5, K=3, batch_size=64, epochs=100, rn=100, rn1=100, eps=5e-5, 
                mu_lr=1e-4, objective_function_values=[], power_constraint_values=[], 
                loss_values=[], mu_k_values=[], baseline=0, mark=0, train=True, num_channels=None,
                num_power_levels=None, power_levels=None):
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
    RESULTS_ROOT = os.path.join(BASE_DIR, '..', 'results_v1')  # Ajusta si tu estructura cambia

    # Y reemplaza toda la lógica de creación de 'path' con esto:
    subfolder = f"{band[b5g]}_{building_id}/torch_results/n_layers{num_layers}_order{K}"
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

    # =========================================================================
    # TRAINING METRICS PLOTS
    # =========================================================================
    
    if len(objective_function_values) > 0:
        plt.figure(figsize=(16, 9))
        plt.title('Función Objetivo (Sum Rate)', fontsize=16, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=14)
        plt.ylabel('Sum Rate', fontsize=14)
        plt.plot(objective_function_values, linewidth=2, color='blue')
        plt.grid(alpha=0.3)
        image_name = f'objective_function_{eps_str}_{mu_lr_str}_{batch_size_str}.png'
        plt.savefig(os.path.join(path, image_name), dpi=150, bbox_inches='tight')
        plt.close()

    if len(power_constraint_values) > 0:
        plt.figure(figsize=(16, 9))
        plt.title('Restricción de Potencia', fontsize=16, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=14)
        plt.ylabel('Violación de Restricción', fontsize=14)
        plt.plot(power_constraint_values, linewidth=2, color='red')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Objetivo (≤0)')
        plt.ylim(-2, 1)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        image_name = f'power_constraint_{eps_str}_{mu_lr_str}_{batch_size_str}.png'
        plt.savefig(os.path.join(path, image_name), dpi=150, bbox_inches='tight')
        plt.close()

    if len(loss_values) > 0:
        plt.figure(figsize=(16, 9))
        plt.title('Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.plot(loss_values, linewidth=2, color='green')
        plt.grid(alpha=0.3)
        image_name = f'loss_{eps_str}_{mu_lr_str}_{batch_size_str}.png'
        plt.savefig(os.path.join(path, image_name), dpi=150, bbox_inches='tight')
        plt.close()

    if len(mu_k_values) > 0:
        plt.figure(figsize=(16, 9))
        plt.title('Multiplicadores de Lagrange (μ_k)', fontsize=16, fontweight='bold')
        plt.xlabel('Iteraciones (x10)', fontsize=14)
        plt.ylabel('μ_k', fontsize=14)
        
        mu_k_array = np.array(mu_k_values)
        if mu_k_array.ndim == 1:
            plt.plot(mu_k_array, linewidth=2, label='Todos los links')
        else:
            for link_idx in range(mu_k_array.shape[1]):
                plt.plot(mu_k_array[:, link_idx], linewidth=2, label=f'Link {link_idx}')
        
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        image_name = f'mu_k_{eps_str}_{mu_lr_str}_{batch_size_str}.png'
        plt.savefig(os.path.join(path, image_name), dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # POLICY ANALYSIS (CORRECTED)
    # =========================================================================
    
    if len(normalized_psi_values) > 0:
        normalized_psi_array = np.array(normalized_psi_values)  # [iterations, num_links, num_actions]
        
        num_links = normalized_psi_array.shape[1]
        num_actions = normalized_psi_array.shape[2]
        num_channels_actual = num_actions - 1  # Restamos "no TX"
        
        print(f"\nShape de probabilidades: {normalized_psi_array.shape}")
        print(f"  - Iteraciones guardadas: {normalized_psi_array.shape[0]}")
        print(f"  - Número de links: {num_links}")
        print(f"  - Número de acciones: {num_actions} (1 no TX + {num_channels_actual} canales)")
        
        # Crear carpeta para políticas por link
        policy_dir = os.path.join(path, 'policy_per_link')
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
        
        # =====================================================================
        # PLOT 1: Evolución de política por link (individual)
        # =====================================================================
        
        for link_idx in range(num_links):
            fig, ax1 = plt.subplots(1, 1, figsize=(16, 13))            
            # Subplot 1: Todas las acciones
            ax1.set_title(f'Link {link_idx}: Evolución de Probabilidades de Acción', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Iteraciones (x10)', fontsize=12)
            ax1.set_ylabel('Probabilidad', fontsize=12)
            
            # Plotear cada acción
            for action_idx in range(num_actions):
                if action_idx == 0:
                    label = 'No TX'
                    color = 'black'
                    linestyle = '--'
                    linewidth = 2.5
                else:
                    channel = action_idx - 1
                    label = f'Canal {channel}'
                    color = f'C{channel}'
                    linestyle = '-'
                    linewidth = 2
                
                ax1.plot(normalized_psi_array[:, link_idx, action_idx], 
                        label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            
            ax1.legend(fontsize=11, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.05, 1.05)
            
            
            plt.tight_layout()
            image_name = f'link_{link_idx}_policy_evolution.pdf'
            plt.savefig(os.path.join(policy_dir, image_name), dpi=150, bbox_inches='tight')

            image_name = f'link_{link_idx}_policy_evolution.svg'
            plt.savefig(os.path.join(policy_dir, image_name), dpi=150, bbox_inches='tight')
            plt.close()
        
        # =====================================================================
        # PLOT 2: Todas las políticas juntas (overview)
        # =====================================================================
        
        fig, axes = plt.subplots(num_links, 1, figsize=(16, 5*num_links))
        if num_links == 1:
            axes = [axes]
        
        for link_idx in range(num_links):
            ax = axes[link_idx]
            
            for action_idx in range(num_actions):
                if action_idx == 0:
                    label = 'No TX'
                    color = 'black'
                    linestyle = '--'
                    linewidth = 2
                else:
                    channel = action_idx - 1
                    label = f'Canal {channel}'
                    color = f'C{channel}'
                    linestyle = '-'
                    linewidth = 2
                
                ax.plot(normalized_psi_array[:, link_idx, action_idx], 
                       label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            
            ax.set_title(f'Link {link_idx}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Iteraciones (x10)', fontsize=11)
            ax.set_ylabel('Probabilidad', fontsize=11)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'all_links_policies.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # PLOT 3: Heatmap de probabilidades finales
        # =====================================================================
        
        final_probs = normalized_psi_array[-1]  # [num_links, num_actions]
        
        fig, ax = plt.subplots(figsize=(12, max(6, num_links*1.5)))
        
        im = ax.imshow(final_probs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Etiquetas
        action_labels = ['No TX'] + [f'Ch {i}' for i in range(num_channels_actual)]
        ax.set_xticks(np.arange(num_actions))
        ax.set_yticks(np.arange(num_links))
        ax.set_xticklabels(action_labels, fontsize=11)
        ax.set_yticklabels([f'Link {i}' for i in range(num_links)], fontsize=11)
        
        # Añadir valores en cada celda
        for i in range(num_links):
            for j in range(num_actions):
                text_color = 'white' if final_probs[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{final_probs[i, j]:.3f}',
                       ha="center", va="center", 
                       color=text_color,
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Acción', fontsize=13, fontweight='bold')
        ax.set_ylabel('Link', fontsize=13, fontweight='bold')
        ax.set_title('Probabilidades Finales de Acción (Última Iteración)', 
                    fontsize=15, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Probabilidad')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'final_probabilities_heatmap.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # PLOT 4: Convergencia de acciones
        # =====================================================================
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        iterations = np.arange(len(normalized_psi_array))
        
        for link_idx in range(num_links):
            # Calcular acción más probable en cada iteración
            most_likely_actions = np.argmax(normalized_psi_array[:, link_idx, :], axis=1)
            
            ax.plot(iterations, most_likely_actions,
                   label=f'Link {link_idx}',
                   linewidth=2.5,
                   marker='o',
                   markersize=5)
        
        ax.set_xlabel('Iteraciones (x10)', fontsize=13)
        ax.set_ylabel('Acción Más Probable', fontsize=13)
        ax.set_yticks(range(num_actions))
        ax.set_yticklabels(['No TX'] + [f'Ch {i}' for i in range(num_channels_actual)])
        ax.set_title('Convergencia: Acción Preferida por Link', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'action_convergence.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # =====================================================================
        # ANÁLISIS FINAL (en consola y archivo de texto)
        # =====================================================================
        
        analysis_file = os.path.join(path, 'policy_analysis.txt')
        with open(analysis_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ANÁLISIS DE POLÍTICA FINAL\n")
            f.write("="*70 + "\n\n")
            
            print("\n" + "="*70)
            print("ANÁLISIS DE POLÍTICA FINAL")
            print("="*70)
            
            for link_idx in range(num_links):
                probs = final_probs[link_idx, :]
                best_action = np.argmax(probs)
                
                if best_action == 0:
                    action_name = "No TX"
                else:
                    action_name = f"Canal {best_action - 1}"
                
                # Escribir en archivo
                f.write(f"\nLink {link_idx}:\n")
                f.write(f"  Acción preferida: {action_name} (p={probs[best_action]:.4f})\n")
                f.write(f"  Distribución completa:\n")
                
                # Imprimir en consola
                print(f"\nLink {link_idx}:")
                print(f"  Acción preferida: {action_name} (p={probs[best_action]:.4f})")
                print(f"  Distribución completa:")
                
                for action_idx in range(num_actions):
                    if action_idx == 0:
                        name = "No TX   "
                    else:
                        name = f"Canal {action_idx-1}"
                    
                    bar = '█' * int(probs[action_idx] * 50)
                    line = f"    {name:8s}: {probs[action_idx]:.4f} {bar}"
                    
                    f.write(line + "\n")
                    print(line)
            
            # Análisis de conflictos
            f.write("\n" + "-"*70 + "\n")
            f.write("ANÁLISIS DE CONFLICTOS DE CANAL\n")
            f.write("-"*70 + "\n")
            
            print("\n" + "-"*70)
            print("ANÁLISIS DE CONFLICTOS DE CANAL")
            print("-"*70)
            
            for channel in range(num_channels_actual):
                action_idx = channel + 1
                links_using_ch = []
                
                for link_idx in range(num_links):
                    if final_probs[link_idx, action_idx] > 0.3:  # Umbral
                        links_using_ch.append((link_idx, final_probs[link_idx, action_idx]))
                
                if len(links_using_ch) > 1:
                    msg = f"  ⚠ Canal {channel}: Usado por {len(links_using_ch)} links (INTERFERENCIA)"
                    for link_id, prob in links_using_ch:
                        msg += f"\n      Link {link_id} (p={prob:.3f})"
                elif len(links_using_ch) == 1:
                    link_id, prob = links_using_ch[0]
                    msg = f"  ✓ Canal {channel}: Usado solo por Link {link_id} (p={prob:.3f})"
                else:
                    msg = f"  - Canal {channel}: No usado"
                
                f.write(msg + "\n")
                print(msg)
            
            f.write("\n" + "="*70 + "\n")
            print("="*70)
            print(f"\nAnálisis completo guardado en: {analysis_file}")

    # =========================================================================
    # SAVE FINAL NORMALIZED PSI
    # =========================================================================
    
    normalized_psi = torch.mean(normalized_psi, dim=0)
    normalized_psi_array = normalized_psi.detach().numpy()
    psi_path = os.path.join(path, f'normalized_psi_{eps_str}_{mu_lr_str}_{batch_size_str}.txt')
    np.savetxt(psi_path, normalized_psi_array, delimiter=',', fmt='%.4f')

    return path