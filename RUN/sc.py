import pickle
import numpy as np
import os
from networks import build_adhoc_network, build_line_network, build_cellular_network, build_adhoc_network_sc

num_graphs = 64000
num_links = 6
pl = 1e-5

graphs = []
for _ in range(num_graphs):
    pl = np.random.rand()
    L = build_adhoc_network(num_links, pl)
    graphs.append(L)

# Guarda el archivo en la ruta esperada por tu función
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, '..', 'graphs', '2_4_990')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'sc_graphs.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(graphs, f)

print(f"Guardados {num_graphs} grafos sintéticos de {num_links} linkscle en {output_path}")