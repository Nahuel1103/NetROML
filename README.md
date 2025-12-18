# 📶 Wireless Learning Toolkit

Herramientas para procesar datos de redes inalámbricas, generar grafos y analizar coeficientes de canal. Diseñado para datos del Plan Ceibal.

> **⚠️ Importante 1:** Los datos_ceibal y la carpeta graphs no son parte del repositorio pero se incluyen con la siguiente estructura a modo de ejemplo.
---

> **⚠️ Importante 2:** Para mayor orden se sugiere sacar los scripts que hay en la carpeta graphs a una carpeta `preprocesamiento` y agregar el archivo `utils.py` a esa carpeta
--- 

---

## 📋 Menú
- [📂 Contenido del repositorio](#-contenido-del-repositorio)
- [⚙️ Configuración](#️-configuración)
- [📁 Estructura de carpetas](#-estructura-de-carpetas)
- [📝 Scripts clave](#-scripts-clave)

---

## 📂 Contenido del repositorio y más
- **`datos_ceibal/`**: Datos brutos en formato `.csv.gz` y `.tgz` (RSSI, MACs, APs, CLIENTES).
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
- `baseline_train.py`: Entrenamiento de línea base con políticas fijas.
- `train.py`: Entrenamiento del modelo GNN principal.
- `gnn.py`: Implementación de la arquitectura GNN para optimización.
- `plot_rates.py`: Compara métricas entre modelos entrenados.
- `plot_results_torch.py`: Genera gráficos de entrenamiento (loss, función objetivo).
- `utils.py`: Funciones auxiliares (transformación de matrices, cálculo de tasas, restricciones).

---

## ⚙️ Configuración
1. **Requisitos**:
   ```bash
   pip install pandas numpy networkx matplotlib scipy torch torch-geometric pickle seaborn textwrap
   ```
2. **Datos APs**:
   - Colocar archivos `.tgz` en `datos_ceibal/datos_resto_del_anio/`.
3. **Datos Clientes**:
   - Colocar archivos `.csv.gz` en `datos_ceibal/datos_ceibal_clientes/`.

## 📁 Estructura sugerida de carpetas:
```
.
├── datos_ceibal/
│   └── datos_resto_del_anio/
│       ├── datos_APs_Abril.csv.gz
│       ├── datos_APs_Agosto.csv.gz
│       └── ...
│   └── datos_ceibal_clientes/
│       ├── RSSI_WLCs_2018-02-01_11_05.tgz
│       ├── RSSI_WLCs_2018-02-01_15_05
│       └── ...
├── Exploración_del_dataset/
│   ├── Expl_dataset.ipynb
│   └── ...
├── preprocesamiento/
│   ├── building_id_count.py
│   └── ...
├── graphs/
│   ├── 5_990/                   # Banda 5GHz, Edificio 990
│   │   ├── train_5_graphs_990.pkl
│   │   ├── val_5_graphs_990.pkl
│   │   └── channel_matrix_coefficients_990/
│   └── ...
├── gnn.py
├── plot_rates.py
├── plot_results_torch.py
├── baseline_train.py
├── train.py
├── utils.py
├── README.md
```

---

## 📝 Scripts clave
| Script | Función |
|--------|---------|
| `gnn.py` | Define la arquitectura GNN para optimizar políticas de potencia. |
| `train.py` | Entrena el modelo GNN con actualización de multiplicadores de Lagrange. |
| `baseline_train.py` | Implementa políticas fijas (ej: potencia uniforme) para comparación. |
| `plot_results_torch.py` | Genera gráficos de métricas (loss, restricciones, función objetivo). |
| `plot_rates.py` | Compara tasas de transmisión entre modelos. |
| `utils.py` | Funciones para transformar matrices de canal, calcular tasas y restricciones. |
| `process_ceibal_data.py` | Genera grafos por mes (1 `.pkl`/mes). |
| `join_graphs.py` | Combina meses en `train.pkl` y `val.pkl`. |
