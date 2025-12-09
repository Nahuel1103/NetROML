import pickle
import numpy as np
import os

# Create dummy data
num_graphs = 100
n_APs = 5
graphs = []

for _ in range(num_graphs):
    # Random channel matrix
    H = np.random.rand(n_APs, n_APs).astype(np.float32)
    graphs.append(H)

# Save to file
path = '/home/bruno/Proyecto/NetROML/RUN/Bruno/preprod/data/'
os.makedirs(path, exist_ok=True)
file_name = 'synthetic_graphs.pkl'

with open(path + file_name, 'wb') as f:
    pickle.dump(graphs, f)

print(f"Dummy data saved to {path + file_name}")
