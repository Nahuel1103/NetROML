from torch_geometric.nn import LayerNorm, Sequential
from torch_geometric.nn.conv import MessagePassing

import random
import pickle
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TAGConv
from torch_geometric.nn import GCNConv
import os
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io

from plot_results_torch import plot_results
from gnn import GNN
from utils import graphs_to_tensor
from utils import get_gnn_inputs
from utils import power_constraint_per_ap
from utils import objective_function
from utils import mu_update
from utils import nuevo_get_rates

from utils import graphs_to_tensor_synthetic

def plot_exploration_stats(exploration_stats, batch_idx, epoch, num_channels, num_power_levels):
    """
    Grafica las estadísticas de exploración para cada link
    """
    total_combinations = num_channels * num_power_levels
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Exploración por Link - Epoch {epoch}, Batch {batch_idx}', fontsize=16)
    
    # 1. Gráfico de cobertura por link
    coverage_data = []
    link_names = []
    for link_idx in range(len(exploration_stats)):
        stats = exploration_stats[link_idx]
        coverage = len(stats['combinations_seen']) / total_combinations * 100
        coverage_data.append(coverage)
        link_names.append(f'Link {link_idx}')
    
    axes[0, 0].bar(link_names, coverage_data, color=['red' if x < 50 else 'orange' if x < 80 else 'green' for x in coverage_data])
    axes[0, 0].set_title('Cobertura de Exploración (%)')
    axes[0, 0].set_ylabel('Porcentaje')
    axes[0, 0].set_ylim(0, 100)
    for i, v in enumerate(coverage_data):
        axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # 2. Heatmap de combinaciones exploradas por link
    exploration_matrix = np.zeros((len(exploration_stats), num_channels * num_power_levels))
    
    for link_idx, stats in exploration_stats.items():
        for combo in stats['combinations_seen']:
            combo_idx = combo[0] * num_power_levels + combo[1]  # Mapear (canal, potencia) a índice
            exploration_matrix[link_idx, combo_idx] = 1
    
    im = axes[0, 1].imshow(exploration_matrix, cmap='RdYlGn', aspect='auto')
    axes[0, 1].set_title('Matrix de Combinaciones Exploradas')
    axes[0, 1].set_xlabel('Combinaciones (Canal, Potencia)')
    axes[0, 1].set_ylabel('Links')
    axes[0, 1].set_yticks(range(len(exploration_stats)))
    axes[0, 1].set_yticklabels([f'Link {i}' for i in range(len(exploration_stats))])
    
    # Etiquetas del eje x
    combo_labels = [f'C{c}P{p}' for c in range(num_channels) for p in range(num_power_levels)]
    axes[0, 1].set_xticks(range(len(combo_labels)))
    axes[0, 1].set_xticklabels(combo_labels, rotation=45, fontsize=8)
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Gráfico de canales únicos por link
    channels_count = [len(stats['channels_used']) for stats in exploration_stats.values()]
    axes[0, 2].bar(link_names, channels_count, color='skyblue')
    axes[0, 2].set_title('Canales Únicos Explorados')
    axes[0, 2].set_ylabel('Número de Canales')
    axes[0, 2].set_ylim(0, num_channels)
    for i, v in enumerate(channels_count):
        axes[0, 2].text(i, v + 0.1, str(v), ha='center')
    
    # 4. Gráfico de potencias únicas por link
    powers_count = [len(stats['powers_used']) for stats in exploration_stats.values()]
    axes[1, 0].bar(link_names, powers_count, color='lightcoral')
    axes[1, 0].set_title('Niveles de Potencia Explorados')
    axes[1, 0].set_ylabel('Número de Potencias')
    axes[1, 0].set_ylim(0, num_power_levels)
    for i, v in enumerate(powers_count):
        axes[1, 0].text(i, v + 0.1, str(v), ha='center')
    
    # 5. Detalles de combinaciones por link (texto)
    axes[1, 1].axis('off')
    text_info = "Combinaciones por Link:\n"
    for link_idx, stats in exploration_stats.items():
        combinations = sorted(stats['combinations_seen'])
        text_info += f"Link {link_idx}: {combinations}\n"
    axes[1, 1].text(0.1, 0.9, text_info, transform=axes[1, 1].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # 6. Resumen estadístico
    axes[1, 2].axis('off')
    summary_text = f"Resumen:\n"
    summary_text += f"Total combinaciones posibles: {total_combinations}\n"
    summary_text += f"Links con exploración completa (100%): {sum(1 for c in coverage_data if c == 100)}\n"
    summary_text += f"Links con buena exploración (>80%): {sum(1 for c in coverage_data if c > 80)}\n"
    summary_text += f"Links problemáticos (<50%): {sum(1 for c in coverage_data if c < 50)}\n"
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=11, verticalalignment='top')
    
    plt.tight_layout()
    return fig


def run(building_id=990, b5g=False, num_links=5, num_channels=5, num_layers=5, K=3,
        batch_size=64, epochs=100, eps=5e-5, mu_lr=1e-4, synthetic=1, rn=100, rn1=100,
        p0=4, sigma=1e-4):   

    banda = ['2_4', '5']
    eps_str = str(f"{eps:.0e}")
    mu_lr_str = str(f"{mu_lr:.0e}")

    if synthetic:
        x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(
            num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset[:7000], batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        x_tensor, channel_matrix_tensor = graphs_to_tensor(
            train=True, num_links=num_links, num_features=1, b5g=b5g, building_id=building_id
        )
        dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    max_antenna_power_dbm=6
    max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
    pmax_per_ap = max_antenna_power_mw*0.7* torch.ones((num_links,))
    mu_k =  torch.ones((num_links,), requires_grad=False)  

    #pmax = 4

    # ---- Definición de red ----
    input_dim = 1
    hidden_dim = 1

    # Definí los niveles de potencia discretos
    power_levels = torch.tensor([max_antenna_power_mw/2, max_antenna_power_mw])
    num_power_levels = len(power_levels)

  
    output_dim = 1 + num_channels*num_power_levels

    dropout = False
    
    gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K)
    optimizer = optim.Adam(gnn_model.parameters(), lr=mu_lr)

    objective_function_values = []
    power_constraint_values = []
    loss_values = []
    mu_k_values = []
    probs_values = []
    power_values = []  


    for epoc in range(epochs):
        print("Epoch number: {}".format(epoc))
        for batch_idx, data in enumerate(dataloader):
    
            channel_matrix_batch = data.matrix
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_links, num_links) # [64, 5, 5]
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)   # [batch*num_links, num_actions]
            psi = psi.view(batch_size, num_links, output_dim)                  # [batch, num_links, num_actions]
          
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            psi = psi.view(batch_size, num_links, output_dim)
            action_logits = psi    
            action_probs = torch.softmax(action_logits, dim=-1)                    # [batch, links, output_dim]
            action_dist = torch.distributions.Categorical(probs=action_probs)
            actions = action_dist.sample()   
            log_p_actions = action_dist.log_prob(actions)                          # [batch, links]

            # Mascara de transmisión (acción != 0)
            transmit_mask = (actions != 0)
            # Log probability total por muestra (sumamos sobre links)
            log_p_sum = log_p_actions.sum(dim=1).unsqueeze(-1)                     # [batch, 1]

            # Construir phi (potencias por canal)
            phi = torch.zeros(batch_size, num_links, num_channels)

            # Solo asignar potencia si decide transmitir
            transmit_indices = torch.where(transmit_mask)
            if len(transmit_indices[0]) > 0:
                batch_idx_tx = transmit_indices[0]
                link_idx_tx = transmit_indices[1]
                selected_actions = actions[batch_idx_tx, link_idx_tx]    
                         # valores 1..(C*P)
                # convertir a canal y potencia
                selected_minus1 = selected_actions - 1
                selected_channels = (selected_minus1 // num_power_levels).long()
                selected_powers_idx = (selected_minus1 % num_power_levels).long()
                # Obtener valores reales de potencia (ej. en mW)
                selected_powers_values = power_levels[selected_powers_idx].to(phi.dtype)

                # Asignar phi
                phi[batch_idx_tx, link_idx_tx, selected_channels] = selected_powers_values

########################################################################################################################

                # # Diagnóstico de exploración por link
                # exploration_stats = {}

                # for link_idx in range(num_links):
                #     exploration_stats[link_idx] = {
                #         'combinations_seen': set(),
                #         'channels_used': set(),
                #         'powers_used': set()
                #     }

                # # Recorrer todas las asignaciones de este batch
                # transmit_indices = torch.where(transmit_mask)
                # if len(transmit_indices[0]) > 0:
                #     batch_idx_tx = transmit_indices[0]
                #     link_idx_tx = transmit_indices[1]
                    
                #     for i in range(len(batch_idx_tx)):
                #         batch_i = batch_idx_tx[i].item()
                #         link_i = link_idx_tx[i].item()
                        
                #         # Encontrar qué canal y potencia se asignó
                #         phi_link = phi[batch_i, link_i, :]  # [num_channels]
                #         active_channel = torch.where(phi_link > 0)[0]
                        
                #         if len(active_channel) > 0:
                #             channel = active_channel[0].item()
                #             power_value = phi_link[channel].item()
                            
                #             # Encontrar índice de potencia
                #             power_idx = None
                #             for p_idx, p_val in enumerate(power_levels):
                #                 if abs(power_value - p_val) < 1e-6:
                #                     power_idx = p_idx
                #                     break
                            
                #             if power_idx is not None:
                #                 combination = (channel, power_idx)
                #                 exploration_stats[link_i]['combinations_seen'].add(combination)
                #                 exploration_stats[link_i]['channels_used'].add(channel)
                #                 exploration_stats[link_i]['powers_used'].add(power_idx)

                # # Imprimir estadísticas y crear gráficos cada cierto número de batches
                # if batch_idx % 50 == 0:
                #     print(f"\n=== Batch {batch_idx} - Estadísticas de Exploración ===")
                #     total_combinations = num_channels * num_power_levels
                    
                #     # Print original (mantener)
                #     for link_idx in range(num_links):
                #         stats = exploration_stats[link_idx]
                #         combinations_seen = len(stats['combinations_seen'])
                #         coverage = combinations_seen / total_combinations * 100
                        
                #         print(f"Link {link_idx}:")
                #         print(f"  Combinaciones vistas: {combinations_seen}/{total_combinations} ({coverage:.1f}%)")
                #         print(f"  Canales usados: {sorted(stats['channels_used'])}")
                #         print(f"  Potencias usadas: {sorted(stats['powers_used'])}")
                #         print(f"  Combinaciones: {sorted(stats['combinations_seen'])}")
                    
                #     # NUEVO: Crear y mostrar gráficos
                #     try:
                #         fig = plot_exploration_stats(exploration_stats, batch_idx, epoc, num_channels, num_power_levels)
                        
                #         # Guardar gráfico (opcional)
                #         filename = f'exploration_epoch_{epoc}_batch_{batch_idx}.png'
                #         fig.savefig(filename, dpi=150, bbox_inches='tight')
                        
                #         # Mostrar gráfico
                #         plt.show()
                #         plt.close()
                        
                #         print(f"Gráfico guardado como: {filename}")
                        
                #     except Exception as e:
                #         print(f"Error creando gráfico: {e}")

########################################################################################################################
            
            # --- constraint de potencia ---
            power_constr_per_ap = power_constraint_per_ap(phi, pmax_per_ap)  # [batch_size, num_links]
            power_constr_per_ap_mean = torch.mean(power_constr_per_ap, dim=(0,1))  
            rates = nuevo_get_rates(phi, channel_matrix_batch, sigma, p0=max_antenna_power_mw)

            sum_rate = -objective_function(rates).unsqueeze(-1) 
            sum_rate_mean = torch.mean(sum_rate, dim=0)

            # Actualización de la penalización
            mu_k = mu_update(mu_k, power_constr_per_ap, eps) 

            penalty_per_ap = power_constr_per_ap * mu_k.unsqueeze(0)  
            total_penalty = penalty_per_ap.sum(dim=1).unsqueeze(-1)   

            cost = sum_rate + total_penalty  
            
            loss = cost * log_p_sum        
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            if batch_idx%10 == 0:
                probs_mean_batch = action_probs.mean(dim=0)  # Promedio sobre batch: [num_links, num_channels+1]
                probs_values.append(probs_mean_batch.detach().numpy())
                              
                # Guardar también información de potencia (phi)
                phi_mean_batch = phi.mean(dim=0)  # Promedio sobre batch: [num_links, num_channels]
                power_values.append(phi_mean_batch.detach().numpy())

                power_constraint_values.append(power_constr_per_ap_mean.detach().numpy())
                objective_function_values.append(-sum_rate_mean.detach().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().numpy())


    for name, param in gnn_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    path = plot_results(
            building_id=building_id,
            b5g=b5g,
            normalized_psi=psi, 
            normalized_psi_values=probs_values,
            power_values=power_values,  # ← AGREGA ESTA LÍNEA
            num_layers=num_layers,
            K=K,
            batch_size=batch_size,
            epochs=epochs,
            rn=rn,
            rn1=rn1,
            eps=eps,
            mu_lr=mu_lr,
            objective_function_values=objective_function_values,
            power_constraint_values=power_constraint_values,
            loss_values=loss_values,
            mu_k_values=mu_k_values,
            num_channels = num_channels,
            num_power_levels = num_power_levels,
            power_levels = power_levels
        )

    file_name = path + 'objective_function_values_train_' + str(epochs) + '.pkl'
    with open(file_name, 'wb') as archivo:
        pickle.dump(objective_function_values, archivo)

    # save trained gnn weights in .pth
    torch.save(gnn_model.state_dict(), path + 'gnn_model_weights.pth')
        
if __name__ == '__main__':
    import argparse

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    rn = 267309
    rn1 = 502321
    torch.manual_seed(rn)
    np.random.seed(rn1) 

    parser = argparse.ArgumentParser(description= 'System configuration')

    parser.add_argument('--building_id', type=int, default=990)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_links', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eps', type=float, default=5e-4)
    parser.add_argument('--mu_lr', type=float, default=5e-4)
    parser.add_argument('--synthetic', type=int, default=0)
    
    args = parser.parse_args()
    
    print(f'building_id: {args.building_id}')
    print(f'b5g: {args.b5g}')
    print(f'num_links: {args.num_links}')
    print(f'num_layers: {args.num_layers}')
    print(f'k: {args.k}')
    print(f'epochs: {args.epochs}')
    print(f'batch_size: {args.batch_size}')
    print(f'eps: {args.eps}')
    print(f'mu_lr: {args.mu_lr}')
    print(f'synthetic: {args.synthetic}')
    
    run(building_id=args.building_id, b5g=args.b5g, num_links=args.num_links, num_layers=args.num_layers, K=args.k, batch_size=args.batch_size, epochs=args.epochs, eps=args.eps, mu_lr=args.mu_lr, synthetic=args.synthetic, rn=rn, rn1=rn1)
    print('Seeds: {} and {}'.format(rn, rn1))