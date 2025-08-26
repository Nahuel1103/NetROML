import random
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

        
def process_raw_ceibal_data(local_n_mac_path, rssi_data_path, b5G = True, building_id = 1190):
    """ Funcion que devuelve una lista de grafos de un mismo edifcio para diferentes instantes de 
        tiempo con datos de ceibal para meses distintos de agosto
    
    Parameters: 
        local_n_mac_path (string): ruta a archivo .csv que mapea building_id con direccion MAC
        rssi_data_path (string): ruta a archivo .csv de datos de RSSI        
        b5G (bool): eleccion de la banda de frecuencia de la transmision. Si True, entonces banda de 5GHz. Si False, entonces banda de 2.4GHz
        building_id (int): eleccion del numero de edificio
        
    Returns: 
        grafos (list): lista donde cada elemento es un grafo correspondiente a un instante de tiempo diferente
        nule_graphs (float): cantidad de grafos nulos encontrados
        valid_graphs (float): cantidad de grafos con la correcta cantidad de nodos encontrados
        total_graphs (float): cantidad de grafos encontrados
    """

    local_n_mac = pd.read_csv(local_n_mac_path)
    df_rssi = pd.read_csv(rssi_data_path)
    random.seed(42)

    local_n_mac_part = local_n_mac[local_n_mac.building_id == building_id]

    df_rssi.drop_duplicates(inplace = True)

    # elimino columnas que no interesan
    df_rssi = df_rssi.drop(['Unnamed: 0', 'MAC_AP', 'Canal',
                        'WLC', 'MAC_Hexa_vecino', 'Canal_vecino', 'MAC_vecino_hexa', 'MAC_vecino', 'RSSI_vecino', 'Tx_power_vecino'], axis = 1)

    # eligo la banda a trabajar
    df_rssi = df_rssi[df_rssi.Banda == b5G]
    df_rssi = df_rssi.drop(['Banda'], axis = 1)

    # elimino los NaNs
    df_rssi.dropna(inplace = True)

    ### uno ambos dataframes ###
    agrupados_por_hora = df_rssi.groupby('timestamp')
    # lista para guardas los grafos
    grafos = []

    # llevo cuenta de cantidad de grafos nulos, cantidad de grafos con la correcta cantidad de nodos y la cantidad de grafos con nodos faltantes
    nule_graphs = 0
    valid_graphs = 0
    total_graphs = 0
    for hora, df_ap_hora in agrupados_por_hora:

        # tomo en cuenta las entradas del df de rssi de ese timestamp solo
        df_rssi_hora = df_rssi[df_rssi['timestamp'] == hora]
    
        # elimino las entradas de nodos que no pertencen al building_id a analizar

        aps_a_mantener = local_n_mac_part['MAC_AP_hexa'].unique()
        # filtrar las filas que cumplen con la condición
        df_rssi_hora = df_rssi_hora[df_rssi_hora['MAC_AP_hexa'].isin(aps_a_mantener)]
        df_rssi_hora = df_rssi_hora[df_rssi_hora['MAC_vecino_hexa_0'].isin(aps_a_mantener)]

        #print("Cantidad de entradas de interes por timeslot: {}".format(df_rssi_hora.shape))

        df_rssi_hora['Atenuacion'] = df_rssi_hora['Atenuacion'].map(lambda power_db: 10**(power_db/10.0))

        if df_rssi_hora.shape[0] != 0:
            # si tengo datos no nulos los transformo en grafo y lo guardo en una lista
            g = nx.from_pandas_edgelist(df_rssi_hora, source='MAC_vecino_hexa_0', target='MAC_AP_hexa', edge_attr='Atenuacion', create_using=nx.DiGraph())
            num_nodes = g.number_of_nodes()
            if local_n_mac_part.shape[0] == num_nodes:
                grafos.append(g)
                valid_graphs += 1
            total_graphs += 1
        else:
            nule_graphs += 1
            total_graphs += 1

    print("cantidad de grafos nulos: {}".format(nule_graphs))
    print("cantidad de grafos validos: {}".format(valid_graphs))
    print("cantidad total de grafos: {}".format(total_graphs))
    
    return grafos, nule_graphs, valid_graphs, total_graphs

def process_august_ceibal_data(local_n_mac_path, rssi_data_path, b5G = True, building_id = 1190):
    """ Funcion que devuelve una lista de grafos de un mismo edifcio para diferentes instantes de 
        tiempo con datos de ceibal del mes de agosto.
    
    Parameters: 
        local_n_mac_path (string): ruta a archivo .csv que mapea building_id con direccion MAC
        rssi_data_path (string): ruta a archivo .csv de datos de RSSI        
        b5G (bool): eleccion de la banda de frecuencia de la transmision. Si True, entonces banda de 5GHz. Si False, entonces banda de 2.4GHz
        building_id (int): eleccion del numero de edificio
        
    Returns: 
        grafos (list): lista donde cada elemento es un grafo correspondiente a un instante de tiempo diferente
        nule_graphs (float): cantidad de grafos nulos encontrados
        valid_graphs (float): cantidad de grafos con la correcta cantidad de nodos encontrados
        total_graphs (float): cantidad de grafos encontrados
    """

    local_n_mac = pd.read_csv(local_n_mac_path)
    df_rssi = pd.read_csv(rssi_data_path)
    random.seed(42)

    local_n_mac_part = local_n_mac[local_n_mac.building_id == building_id]

    df_rssi.drop_duplicates(inplace = True)

    # elimino columnas que no interesan
    df_rssi = df_rssi.drop(['Unnamed: 0', 'MAC_AP', 'LOCAL', 'MAC_vecino',
                           'AP_name', 'WLC', 'MAC_Hexa_vecino', 'Canal_vecino', 'MAC_vecino_hexa', 'LOCAL_vecino'], axis = 1)

    # eligo la banda a trabajar
    df_rssi = df_rssi[df_rssi.Banda == b5G]
    df_rssi = df_rssi.drop(['Banda'], axis = 1)

    # elimino los NaNs
    df_rssi.dropna(inplace = True)

    df_rssi['Atenuacion'] = df_rssi['RSSI_vecino'].copy()

    df_rssi['Atenuacion'] = df_rssi['RSSI_vecino'] - df_rssi['Tx_power_vecino']
    # elimino columnas que no interesan
    df_rssi = df_rssi.drop(['RSSI_vecino', 'Tx_power_vecino'], axis = 1)

    ### uno ambos dataframes ###
    agrupados_por_hora = df_rssi.groupby('timestamp')
    # lista para guardas los grafos
    grafos = []

    # recorro df de APs agrupados por timestamp

    # llevo cuenta de cantidad de grafos nulos, cantidad de grafos con la correcta cantidad de nodos y la cantidad de grafos con nodos faltantes
    nule_graphs = 0
    valid_graphs = 0
    total_graphs = 0
    for hora, df_ap_hora in agrupados_por_hora:

        # tomo en cuenta las entradas del df de rssi de ese timestamp solo
        df_rssi_hora = df_rssi[df_rssi['timestamp'] == hora]
    
        # elimino las entradas de nodos que no pertencen al building_id a analizar
        aps_a_mantener = local_n_mac_part['MAC_AP_hexa'].unique()
    
        # filtrar las filas que cumplen con la condición
        df_rssi_hora = df_rssi_hora[df_rssi_hora['MAC_AP_hexa'].isin(aps_a_mantener)]
        df_rssi_hora = df_rssi_hora[df_rssi_hora['MAC_vecino_hexa_0'].isin(aps_a_mantener)]

        df_rssi_hora['Atenuacion'] = df_rssi_hora['Atenuacion'].map(lambda power_db: 10**(power_db/10.0))

        if df_rssi_hora.shape[0] != 0:
            # si tengo datos no nulos los transformo en grafo y lo guardo en una lista
            g = nx.from_pandas_edgelist(df_rssi_hora, source='MAC_vecino_hexa_0', target='MAC_AP_hexa', edge_attr='Atenuacion', create_using=nx.DiGraph())
            num_nodes = g.number_of_nodes()
            if local_n_mac_part.shape[0] == num_nodes:
                grafos.append(g)
                valid_graphs += 1
            total_graphs += 1
        else:
            nule_graphs += 1
            total_graphs += 1
        
    print("cantidad de grafos nulos: {}".format(nule_graphs))
    print("cantidad de grafos validos: {}".format(valid_graphs))
    print("cantidad total de grafos: {}".format(total_graphs))
            
    return grafos, nule_graphs, valid_graphs, total_graphs


if __name__ == '__main__':

    num_grafo = 2
    extern_volume = True
    buildings = [[32, False], [165, False], [307, False], [293, False]]
    banda = ['2_4', '5']

    # si cargo los datos desde USB externo o desde disco
    if extern_volume:
        dir_general = "../../datos_ceibal/datos_resto_del_anio/"
    else:
        dir_general = "../../datos_ceibal/datos_resto_del_anio/"

    #local_n_mac = "./from_mac_to_building_id.csv"
    # cargo el df con mac en hexa
    local_n_mac = "./from_mac_hexa_to_building_id.csv"  

    for building_id, b5g in buildings:

        # creo dataframe solo con columnas para ir guardando las estadisticas
        columnas = ['month', 'nule_graphs', 'valid_graphs', 'total_graphs']
        df = pd.DataFrame(columns=columnas)

        # si no se encuentra creada, creo la carpeta
        folder_name = './' + str(banda[b5g]) + '_' + str(building_id)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name) 

        # paths a archivos .csv de todos los meses excepto agosto (enero y febrero no hay)
        # rel_paths = ['marzo/datos_vecinos_Marzo.csv', 'abril/datos_vecinos_Abril.csv', 'mayo/datos_vecinos_Mayo.csv',
        #            'junio/datos_vecinos_Junio.csv', 'julio/datos_vecinos_Julio.csv', 'agosto/datos_filtrados_vecinos_Agosto.csv', 'setiembre/datos_vecinos_Setiembre.csv',
        #             'octubre/datos_vecinos_Octubre.csv', 'noviembre/datos_vecinos_Noviembre.csv', 'diciembre/datos_vecinos_Diciembre.csv']
        
        rel_paths = ['marzo/datos_vecinos_Marzo.csv']
        
        for path in rel_paths:
            # concateno strings
            complete_path = dir_general + path
            # analizo de forma diferente los datos de agosto
            if path == 'agosto/datos_filtrados_vecinos_Agosto.csv':
                grafos, nule_graph, valid_graph, total_graph = process_august_ceibal_data(local_n_mac, complete_path, b5G = b5g, building_id = building_id)
            else:
                grafos, nule_graph, valid_graph, total_graph = process_raw_ceibal_data(local_n_mac, complete_path, b5G = b5g, building_id = building_id)    
            directorio_archivo = os.path.basename(os.path.dirname(complete_path))
            print('{} procesado'.format(directorio_archivo))
            # agrego entrada al dataframe
            month = directorio_archivo
            nuevo_mes = {'month': month, 'nule_graphs': nule_graph, 'valid_graphs': valid_graph, 'total_graphs': total_graph}
            # agrego la nueva fila al df
            df.loc[len(df)] = nuevo_mes
        
            # guardar la lista en un archivo
            file_name = os.path.join(folder_name, 'list_' + str(banda[b5g]) + '_graphs_' + str(building_id) + '_' + directorio_archivo + '.pkl')
            with open(file_name, 'wb') as file:
                pickle.dump(grafos, file)
        
            if len(grafos) > num_grafo:
                # guardo un ploteo de un grafo para cada mes
                plt.figure(figsize= (16,9))
                pos = nx.spring_layout(grafos[num_grafo])
                nx.draw(grafos[num_grafo], with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', arrowsize=20)
                image_name = os.path.join(folder_name, 'sample_' + str(banda[b5g]) + '_graphs_' + str(building_id) + '_' + directorio_archivo + '.png')
                plt.savefig(image_name)
                plt.close()
            else:
                print('La lista del mes {} es vacia'.format(directorio_archivo))
        # guardo el dataframe como un .csv
        df_name = './' + folder_name + '/' + str(banda[b5g]) + '_' + str(building_id) + '.csv'
        df.to_csv(df_name, index=False)