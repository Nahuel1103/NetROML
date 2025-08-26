import numpy as np
import pandas as pd
import gzip


with gzip.open('../../datos_ceibal/datos_resto_del_anio/datos_filtrados_APs_Agosto.csv.gz', 'rt') as f:
    df_aps = pd.read_csv(f)  # Leer solo las primeras 1000 filas

# df_aps = pd.read_csv('../../datos_ceibal/datos_resto_del_anio/agosto/datos_filtrados_APs_Agosto.csv')

# elimino columnas que no interesan
local_n_mac = df_aps.drop(['Unnamed: 0', 'timestamp', 'MAC_AP_hexa',
                           'AP_name', 'WLC', 'Banda', 'Canal', 'Power_level', 'Modelo', 'Tx_power'], axis = 1)

# elimino los NaNs
local_n_mac.dropna(inplace = True)

# elimino filas duplicadas
local_n_mac.drop_duplicates(inplace = True)

# agrego columna nueva 'building_id'
local_n_mac['building_id'] = local_n_mac['MAC_AP'].copy()

# creo una lista con los valores de la columna LOCAL
local_list = local_n_mac['LOCAL'].to_list()

# elimino los repetidos 
local_list = list(set(local_list))

# funcion para utilizar en .apply() -> mapear valor de LOCAL al índice de la lista
def map_to_index(row, local_list):
    local_id = row['LOCAL']
    indice = local_list.index(local_id)
    return indice + 1

local_n_mac['building_id'] = local_n_mac.apply(map_to_index, local_list=local_list, axis='columns')
local_n_mac.drop(['LOCAL'], axis = 1, inplace = True)

local_n_mac.to_csv('./from_mac_to_building_id.csv', index=False)