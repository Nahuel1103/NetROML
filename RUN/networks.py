import numpy as np
import scipy.io
import pdb
from scipy import sparse
import pickle


#############################################
############ Generate line network ###############
##############################################
# inputs:
# num_channels - number of users
# pl - pathloss factor
# outputs:
# L - matrix of pathloss fading
def build_line_network(num_channels,pl):

    receivers = np.vstack([np.linspace(-num_channels/2,num_channels/2,num_channels), np.zeros(num_channels)]).transpose()

    transmitters = np.vstack([ receivers[:,0] + np.random.uniform(low=-1,high=1, size=(num_channels,)), np.random.uniform(low=-4,high=4, size=(num_channels,))]).transpose()

    L = np.zeros((num_channels,num_channels))

    for i in np.arange(num_channels):
        for j in np.arange(num_channels):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)


    data = {'T': [], 'R':[]}
    data['T'] = transmitters
    data['R'] = receivers
    scipy.io.savemat("line_net" + str(num_channels) + ".mat", data)
    return L

#############################################
############ Generate multi-cell network ###############
##############################################
# inputs:
# n - number of base stations
# k - number of cellular uses per base station
# pl - pathloss factor
# outputs:
# L - matrix of pathloss fading
# assign - kx1 vector assigning users to base station
def build_cellular_network(n, k, pl):
    M = n*k

    n_rows = np.floor(n/3)

    assign = np.kron(np.arange(n),np.ones((1,k))) # assignment of user to base station

    receivers = np.vstack([np.linspace(-M/2,M/2,n), np.zeros(n)]).transpose()
    boundaries = np.linspace(-3*M/4,3*M/4,n+1)

    transmitters_x = np.zeros((M,1))
    transmitters_y = np.zeros((M,1))
    for i in np.arange(n):
        transmitters_x[i*k:(i+1)*k] = receivers[i,0] + np.random.uniform(low=-10,high=10,size=(k,1))
        transmitters_y[i*k:(i+1)*k] = receivers[i,1] + np.random.uniform(low=-10,high=10,size=(k,1))

    transmitters = np.hstack([transmitters_x, transmitters_y])

    data = {'T': [], 'R':[], 'A':[]}
    data['T'] = transmitters
    data['R'] = receivers
    data['A'] = assign[0].astype(int)
    scipy.io.savemat("cell_net.mat", data)

    L = np.zeros((M,n))

    for i in np.arange(M):
        for j in np.arange(n):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)

    return L, assign[0].astype(int)



def build_adhoc_network(num_channels,pl):

    transmitters = np.random.uniform(low=-num_channels, high=num_channels, size=(num_channels,2))
    receivers = transmitters + np.random.uniform(low=-num_channels/4,high=num_channels/4, size=(num_channels,2))

    L = np.zeros((num_channels,num_channels))

    for i in np.arange(num_channels):
        for j in np.arange(num_channels):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)

    for i in range(L.shape[0]):
        L[i,i] += i + 5 

    data = {'T': [], 'R':[], 'L':[]}
    data['T'] = transmitters
    data['R'] = receivers
    data['A'] = L
    #scipy.io.savemat("pl_net" + str(num_channels) + ".mat", data)

    return L

def build_adhoc_network_variable(num_links, pl, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # TX en círculo con radio variable
        radius = np.random.uniform(2.0, 5.0)
        angles = np.linspace(0, 2*np.pi, num_links, endpoint=False) + np.random.uniform(0, 2*np.pi)
        transmitters = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        
        # RX cerca del centro con dispersión variable
        dispersion = np.random.uniform(0.3, 1.5)
        receivers = np.random.randn(num_links, 2) * dispersion
        
        # Pathloss matrix
        L = np.zeros((num_links, num_links))
        for i in range(num_links):
            for j in range(num_links):
                d = max(np.linalg.norm(transmitters[i] - receivers[j]), 0.1)
                L[i,j] = d**(-pl)
        
        # # Diagonal variable (clave para eigenvalues distintos)
        # L += np.diag(np.random.exponential(3.0, num_links))
        L += np.diag(np.random.uniform(0, 3.0, num_links))
        
        return L