import pandas as pd

# cargo el .csv que genero con create_mac_buildingid_df.py
local_n_mac = pd.read_csv('from_mac_to_building_id.csv')

cant_buildings = local_n_mac['building_id'].value_counts()

building_count = pd.DataFrame({'building_id': cant_buildings.index, 'cantidad': cant_buildings.values})

building_count.to_csv('./building_id_count.csv', index=False)
