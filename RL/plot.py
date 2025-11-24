import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

def plot_channel_allocation(episode_allocations, num_channels, num_power_levels):
    """
    Genera un Heatmap detallado de la asignación de recursos.
    
    Args:
        episode_allocations: Lista de pasos. Cada paso es una lista de tuplas (canal, potencia).
                             Ej: [[(0, 3.98), (-1, 0.0)], ...]
        num_channels: Entero, número total de canales disponibles.
        num_power_levels: Entero, para formatear la etiqueta de texto.
    """
    num_steps = len(episode_allocations)
    num_links = len(episode_allocations[0])
    
    # 1. Preparar matrices de datos
    channel_map = np.full((num_links, num_steps), -1) # Matriz de Canales
    power_map = np.zeros((num_links, num_steps))      # Matriz de Potencias (para texto)
    
    for t, step_data in enumerate(episode_allocations):
        for link_idx, (ch, pow_val) in enumerate(step_data):
            channel_map[link_idx, t] = int(ch)
            power_map[link_idx, t] = pow_val

    # 2. Configurar Mapa de Colores Discreto (Categorical)
    # Definimos colores: El primero para -1 (Apagado), luego uno para cada canal
    # Usamos 'Pastel1' o 'Set2' para que sea agradable a la vista
    base_colors = sns.color_palette("Set2", num_channels)
    colors = [(0.9, 0.9, 0.9)] + base_colors # Gris claro para -1, luego los demás
    cmap = mcolors.ListedColormap(colors)
    
    # Límites para normalizar los colores: de -1.5 a num_channels - 0.5
    # Esto asegura que cada entero caiga en el centro de un color
    boundaries = np.arange(-1.5, num_channels - 0.5 + 1, 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # 3. Crear Anotaciones (Texto dentro de las celdas)
    # Si está apagado (-1), ponemos un punto o vacío. Si no, ponemos la potencia.
    annot_data = np.empty((num_links, num_steps), dtype=object)
    for i in range(num_links):
        for j in range(num_steps):
            ch = channel_map[i, j]
            p_val = power_map[i, j]
            if ch == -1:
                annot_data[i, j] = "" # Celda vacía si está apagado
            else:
                # Formato corto para la potencia (ej: 3.9)
                annot_data[i, j] = f"{p_val:.1f}"

    # 4. Generar el Gráfico
    plt.figure(figsize=(max(10, num_steps * 0.5), 6)) # Ancho dinámico según pasos
    
    ax = sns.heatmap(channel_map, 
                     cmap=cmap, 
                     norm=norm,
                     annot=annot_data, 
                     fmt="",          # Importante para que acepte strings en annot
                     cbar=True,
                     linewidths=0.5, 
                     linecolor='white',
                     cbar_kws={"ticks": np.arange(-1, num_channels)})
    
    # Ajustar la barra de colores para que muestre etiquetas legibles
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(-1, num_channels))
    cbar.set_ticklabels(['OFF'] + [f'Ch {i}' for i in range(num_channels)])

    plt.title("Evolución de Asignación: Color=Canal, Texto=Potencia")
    plt.xlabel("Step (Tiempo)")
    plt.ylabel("ID de Enlace")
    plt.tight_layout()
    plt.show()

# --- Ejemplo de uso rápido para probarlo ---
if __name__ == "__main__":
    # Datos simulados: 5 enlaces, 10 pasos
    # Formato: [(canal, potencia), (canal, potencia)...] por cada paso
    dummy_data = []
    for t in range(10):
        step = []
        for l in range(5):
            # Simulamos que algunos se apagan (-1) y otros cambian de canal
            ch = np.random.choice([-1, 0, 1, 2])
            pow_val = 4.0 if ch >= 0 else 0.0
            step.append((ch, pow_val))
        dummy_data.append(step)

    print("Generando gráfico de prueba...")
    plot_channel_allocation(dummy_data, num_channels=3, num_power_levels=2)