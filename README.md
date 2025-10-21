# 📶 Wireless Learning Toolkit

Herramientas para procesar datos de redes inalámbricas, generar grafos y analizar coeficientes de canal. Diseñado para datos del Plan Ceibal.


---

## 📋 Menú
- [📂 Contenido del repositorio](#-contenido-del-repositorio)
- [⚙️ Configuración](#️-configuración)
- [📁 Estructura de carpetas](#-estructura-de-carpetas)
- [📝 Scripts clave](#-scripts-clave)

---

## 📂 Contenido del repositorio y más
- **`preprocesamiento/`**:
  - `building_id_count.py`: Cuenta la cantidad de AP's por building ID.
  - `create_mac_buildingid_df.py`: Mapeo MAC → ID de edificio.
  - `create_mac_hexa_buildingid_df.py`: Mapeo MAC Hexa→ ID de edificio.
  - `process_ceibal_data.py`: Generación de grafos por mes.
  - `join_graphs.py`: Combina meses en conjuntos train/val.
  - `channel_coeffs.py`: Calcula la atenuación en los grafos.
  - `channel_coeff_print.py`: Printea coeficientes específicos del canal para debuggear.      
  - `load_ceibal_data_functions.py`: Carga funciones.
  - `utils.py`: Funciones auxiliares (transformación de matrices, cálculo de tasas, restricciones).
- **`graphs/`**: Resultados (grafos, coeficientes, estadísticas).
- RUN
   - v1: Archivos de la versión 1 del algoritmo
   - v2: Archivos de la versión 2 del algoritmo
   - `baseline_train.py`: Entrenamiento de línea base con políticas fijas.
   - `train.py`: Entrenamiento del modelo GNN principal.
   - `gnn.py`: Implementación de la arquitectura GNN para optimización.
   - `plot_rates.py`: Compara métricas entre modelos entrenados.
   - `plot_results_torch.py`: Genera gráficos de entrenamiento (loss, función objetivo).
   - `utils.py`: Funciones auxiliares (transformación de matrices, cálculo de tasas, restricciones).
   - `networks.py`: Archivo que crea la estructura de las matrices de canal.
   - `sc.py`: Archivo que crea el array con las matrices de canal.


---

## ⚙️ Configuración
**Requisitos**:
   ```bash
   pip install pandas numpy networkx matplotlib scipy torch torch-geometric pickle seaborn textwrap
   ```
