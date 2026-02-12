import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import torch_geometric as pyg
from utils import graphs_to_tensor, get_gnn_inputs, graphs_to_tensor_synthetic, get_rates, objective_function, power_constraint

class WirelessEnv(gym.Env):
    """
    Custom Gymnasium Environment for Wireless Network Optimization using GNNs.
    This environment simulates a wireless network where an agent (GNN) optimizes power allocation.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                building_id=990, 
                b5g=False, 
                num_links=5, 
                num_features=1, 
                sigma=1e-4,
                synthetic=False, 
                train=True
                ):
        """
        Initialize the Wireless Environment.
        
        Args:
            building_id (int): ID of the building to load data for (default: 990).
            b5g (bool): Flag to use 5GHz band data if True, else 2.4GHz (default: False).
            num_links (int): Number of wireless channels available (default: 5).
            num_features (int): Number of input features per node (default: 1).
            synthetic (bool): If True, use synthetic data; otherwise, use real data (default: False).
            train (bool): If True, load training data; otherwise, load validation data.
        """
        super().__init__()
        
        # Store configuration parameters
        self.building_id = building_id
        self.b5g = b5g
        self.num_links = num_links
        self.synthetic = synthetic
        self.train = train
        
        # Load Data based on the configuration
        if synthetic:
            # Load synthetic graph data using utility function specific for synthetic datasets
            x_tensor, channel_matrix_tensor = graphs_to_tensor_synthetic(num_links, num_features=num_features, b5g=b5g, building_id=building_id)
            # Process raw tensors into GNN-ready Data objects (list of PyG Data objects)
            self.dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
            if train:
                # Limit dataset to 7000 samples for training as per original implementation
                self.dataset = self.dataset[:7000] 
        else:
            # Load real-world graph data
            x_tensor, channel_matrix_tensor = graphs_to_tensor(train=train, num_links=num_links, num_features=num_features, b5g=b5g, building_id=building_id)
            # Process into GNN inputs
            self.dataset = get_gnn_inputs(x_tensor, channel_matrix_tensor)
            
        # Initialize indexing for iterating through the dataset
        self.num_samples = len(self.dataset)
        self.current_idx = 0
        self.indices = np.arange(self.num_samples)
        
        # Shuffle data indices if we are in training mode
        if self.train:
            np.random.shuffle(self.indices)
            
        # Define Action and Observation Spaces
        # Note: Since graph sizes vary, standard Box/Discrete spaces are hard to define strictly.
        # We leave them as None or placeholder for now.
        self.action_space = None 
        self.observation_space = None

        # System Constraints and Parameters
        max_antenna_power_dbm = 6
        self.max_antenna_power_mw = 10 ** (max_antenna_power_dbm / 10)
        self.pmax_per_ap = self.max_antenna_power_mw*0.8* torch.ones((num_links,))
        self.sigma = sigma        # Noise power level

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the next state (next graph in the dataset).
        
        Returns:
            data (Data): The PyG Data object representing the graph state.
            info (dict): Empty info dictionary.
        """
        super().reset(seed=seed)
        
        # Check if we reached end of dataset, assuming cyclic iterator behavior
        if self.current_idx >= self.num_samples:
             self.current_idx = 0
             if self.train:
                 # Reshuffle for the next pass
                 np.random.shuffle(self.indices)
        
        # Select the next graph using the shuffled indices
        data_idx = self.indices[self.current_idx]
        self.data = self.dataset[data_idx]
        self.current_idx += 1
        
        # Return the PyG Data object as the observation
        return self.data, {}

    def step(self, action):
        """
        Execute one action step in the environment.
        
        Args:
            action (Tensor): The power allocation 'phi'. 
                             Shape: (Batch_Size, Num_Nodes, num_links) or (Num_Nodes, num_links).
        
        Returns:
            observation (Data): The current graph data (observation).
            reward (float): The reward (sum rate) achieved.
            terminated (bool): Always True (one-step episode).
            truncated (bool): Always False.
            info (dict): Diagnostic info (rates, constraints).
        """
        # Ensure action is a torch Tensor
        phi = action
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi)
            
        # Retrieve the Channel Matrix from the current data object
        channel_matrix = self.data.matrix
        
        # Helper to ensure dimensions match `get_rates` expectation
        # We check dimensions to handle both batched (Batch, N, C) and single (N, C) inputs
        if phi.dim() == 2:
             phi = phi.unsqueeze(0) # Add batch dim -> (1, N, C)
        
        mat = channel_matrix
        if mat.dim() == 2:
            mat = mat.unsqueeze(0) # Add batch dim -> (1, N, N)
            
        # Handling Batched Input from DataLoader
        # If self.data is part of a Batch (loaded via PyG DataLoader outside), 
        # we need to ensure the channel matrix corresponds to the batch structure.
        # The original logic implies a fixed number of nodes/channels for reshaping.
        if hasattr(self.data, 'batch') and self.data.batch is not None:
             # Assume logic for unstacking/reshaping matrix for batched calculation
             batch_size = phi.shape[0]
             N = self.num_links # Using num_links argument as Node count based on original code usage
             mat = mat.view(batch_size, N, N) # Reshape flat matrix list to (Batch, N, N)

        # Calculate Rates using the utility function
        # rates shape: (Batch, N, 1) or similar
        rates = get_rates(phi, mat, self.sigma, self.max_antenna_power_mw)
        
        # Calculate Power Constraint Violation
        power_constr = power_constraint(phi, self.pmax_per_ap)
        
        # Calculate Objective Function (Sum Rate)
        # Note: objective_function returns negative sum rate (-Rate)
        sum_rate = objective_function(rates) 
        
        # Define Reward: We want to Maximize Rate, so Reward = Positive Sum Rate.
        # Since sum_rate from util is Negative, Reward = -sum_rate
        reward = -sum_rate 
        
        # Construct Info Dictionary with detailed metrics
        info = {
            'power_constr': power_constr, # Constraint violation
            'rates': rates,               # Individual user rates
            'sum_rate': -sum_rate         # Total system rate (positive)
        }
        
        # Episode is done after one step (Contextual Bandit)
        terminated = True 
        truncated = False
        
        return self.data, reward, terminated, truncated, info

    def set_dataloader(self, batch_size=1):
        """
        Placeholder for potentially managing internal dataloaders.
        Currently unused as DataLoader is managed externally in train.py.
        """
        pass
