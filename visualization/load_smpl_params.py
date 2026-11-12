#!/usr/bin/env python3
"""
Script to load and inspect SMPL parameters from a pickle file.
"""

import pickle
import numpy as np
import os


def load_smpl_params(file_path):
    """Load SMPL parameters from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def main():
    """Main function to load and print SMPL parameters."""
    file_path = "smpl_params_frame_10.pkl"

    print(f"Loading SMPL parameters from: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File does not exist at {file_path}")
        return

    # Load the pickle file
    smpl_params = load_smpl_params(file_path)

    if smpl_params is None:
        return

    print("\nFile loaded successfully!")
    print(f"Data type: {type(smpl_params)}")

    if isinstance(smpl_params, dict):
        print(f"Available keys: {list(smpl_params.keys())}")

        print("\n--- Parameter Details ---")
        for key, value in smpl_params.items():
            if isinstance(value, np.ndarray):
                print(f"  - {key}:")
                print(f"    - Shape: {value.shape}")
                print(f"    - Dtype: {value.dtype}")
                # Print first few values for small arrays
                if value.size <= 20:
                    print(f"    - Values: {value}")
                else:
                    print(f"    - First 5 values: {value.flatten()[:5]}")
            else:
                print(f"  - {key}: {value}")
    else:
        print(f"Loaded data is not a dictionary. Data:\n{smpl_params}")


if __name__ == "__main__":
    main()
