import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.append("/Users/mauriciovieirarodriguez/project/NetROML")

from model.network_graph_env import NetworkGraphEnv

def test_environment():
    print("Initializing environment...")
    try:
        data_root = Path("/Users/mauriciovieirarodriguez/project/NetROML/buildings")
        env = NetworkGraphEnv(
            data_root=data_root,
            building_id=990,
            max_timesteps=10,
            arrival_rate=5.0, # High rate to ensure clients exist
            mean_duration=20.0,
            debug=True
        )
        print("Environment initialized.")
        
        print("Resetting environment...")
        obs, info = env.reset(seed=42)
        print("Reset successful.")
        
        print(f"Initial State: {env.num_active_clients} active clients.")
        
        for i in range(5):
            print(f"\nStep {i+1}")
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Reward: {reward:.4f}")
            print(f"Info: {info}")
            
            if info['num_active_clients'] > 0:
                print(f"Mean Rate: {info['mean_rate']:.2f} Mbps")
            else:
                print("No active clients.")
                
        print("\nVerification successful!")
        
    except Exception as e:
        print(f"\nVerification FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment()
