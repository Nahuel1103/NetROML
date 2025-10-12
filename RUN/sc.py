import pickle
import numpy as np
from networks import build_adhoc_network, build_line_network, build_cellular_network, build_adhoc_network_sc

num_graphs = 64000
num_links = 3
pl = 5#1e-5

graphs = []
for _ in range(num_graphs):
    pl = np.random.rand()
    L = build_adhoc_network(num_links, pl)
    graphs.append(L)

# Guarda el archivo en la ruta esperada por tu función
output_path = '/Users/nahuelpineyro/NetROML/graphs/2_4_990/sc_graphs.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(graphs, f)

print(f"Guardados {num_graphs} grafos sintéticos de {num_links} linkscle en {output_path}")