
from utils import load_channel_matrix
import torch
import numpy as np

def test_data_loading():
    print("Testing Training Data Loading...")
    train_iter = load_channel_matrix(
        building_id=990,
        b5g=False,
        num_links=5,
        synthetic=False,
        shuffle=False,
        repeat=False,
        train=True
    )
    
    # Count items in train
    train_count = 0
    try:
        for _ in train_iter:
            train_count += 1
    except Exception as e:
        print(f"Error iterating train: {e}")

    print(f"Train samples: {train_count}")

    print("\nTesting Validation Data Loading...")
    val_iter = load_channel_matrix(
        building_id=990,
        b5g=False,
        num_links=5,
        synthetic=False,
        shuffle=False,
        repeat=False,
        train=False
    )
    
    val_count = 0
    try:
        for _ in val_iter:
            val_count += 1
    except Exception as e:
        print(f"Error iterating val: {e}")

    print(f"Validation samples: {val_count}")

    if train_count > 0 and val_count > 0 and train_count != val_count:
        print("\nSUCCESS: Successfully loaded different datasets for train and validation.")
    elif train_count > 0 and val_count > 0:
         print(f"WARNING: Train and Val counts are equal ({train_count}). Check if files are identical.")
    else:
        print("\nFAILURE: Could not load data.")

if __name__ == "__main__":
    test_data_loading()
