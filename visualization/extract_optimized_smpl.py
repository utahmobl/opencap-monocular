#!/usr/bin/env python3
"""
Script to load and inspect SMPL parameters from an optimized pickle file.
"""

import pickle
import numpy as np
import os
import joblib


def load_optimized_smpl(file_path):
    """Load optimized SMPL parameters from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            # Assuming joblib might be needed for this file as well.
            data = joblib.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file with joblib, trying pickle: {e}")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e2:
            print(f"Error loading file with pickle: {e2}")
            return None


def inspect_data_structure(data, indent=0):
    """Inspects and prints the structure of the loaded data."""
    prefix = "  " * indent
    if data is None:
        return

    if isinstance(data, dict):
        print(f"{prefix}Data is a dictionary with keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"{prefix}--- Key: '{key}' ---")
            print(f"{prefix}  - Type: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"{prefix}  - Shape: {value.shape}")
                print(f"{prefix}  - Dtype: {value.dtype}")
            elif isinstance(value, dict):
                inspect_data_structure(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}  - Length: {len(value)}")
                if len(value) > 0:
                    print(f"{prefix}  - Type of first element: {type(value[0])}")
            elif hasattr(value, "shape"):  # For torch tensors
                print(f"{prefix}  - Shape: {value.shape}")
            else:
                print(f"{prefix}  - Value: {value}")
    elif isinstance(data, list):
        print(f"{prefix}Data is a list.")
        print(f"{prefix}  - Length: {len(data)}")
        if len(data) > 0:
            print(f"{prefix}  - Inspecting first element:")
            inspect_data_structure(data[0], indent + 1)
    else:
        print(f"{prefix}Data is of type: {type(data)}")


def extract_and_save_smpl(data, frame_idx, output_filename):
    """Extracts SMPL parameters for a given frame and saves them."""
    if data is None:
        return

    # The data is nested in a dictionary, usually with '0' as the key
    subject_id = list(data.keys())[0]
    subject_data = data[subject_id]

    print("\nExtracting SMPL parameters...")

    smpl_params = {}

    # Extract parameters for the specified frame
    if "pose" in subject_data:
        # pose is (frames, 72) where the first 3 are global_orient and the rest are body_pose
        pose_data = subject_data["pose"][frame_idx]
        smpl_params["global_orient"] = pose_data[:3]  # Axis-angle
        smpl_params["body_pose"] = pose_data[3:]  # Axis-angle
        print(
            f"Extracted 'global_orient' (shape: {smpl_params['global_orient'].shape}) and 'body_pose' (shape: {smpl_params['body_pose'].shape})"
        )

    if "trans" in subject_data:
        smpl_params["transl"] = subject_data["trans"][frame_idx]
        print(f"Extracted 'transl' (shape: {smpl_params['transl'].shape})")

    if "betas" in subject_data:
        # Betas might be optimized per-frame, so take the instance for the current frame
        smpl_params["betas"] = subject_data["betas"][frame_idx]
        print(
            f"Extracted 'betas' for frame {frame_idx} (shape: {smpl_params['betas'].shape})"
        )

    if not smpl_params:
        print("Could not find expected SMPL keys ('pose', 'trans', 'betas').")
        return

    print(f"\nSaving extracted parameters for frame {frame_idx} to {output_filename}")
    with open(output_filename, "wb") as f:
        pickle.dump(smpl_params, f)
    print("Save complete.")

    # Also print the extracted values
    print("\n--- Extracted Parameter Details ---")
    for key, value in smpl_params.items():
        print(f"  - {key}:")
        print(f"    - Shape: {value.shape}")
        print(f"    - Dtype: {value.dtype}")


def main():
    """Main function to load and inspect optimized SMPL parameters."""
    file_path = "/home/selim/opencap-mono/output/case_fixed_1/subject4/Session0/Cam3/squats1/squats1_optimized.pkl"
    frame_to_extract = 10
    # Save the output inside the visualization folder
    output_filename = (
        f"visualization/optimized_smpl_params_frame_{frame_to_extract}.pkl"
    )

    print(f"Loading optimized SMPL data from: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File does not exist at {file_path}")
        return

    # Load the pickle file
    smpl_data = load_optimized_smpl(file_path)

    if smpl_data is None:
        return

    print("\nFile loaded successfully!")

    # inspect_data_structure(smpl_data)
    extract_and_save_smpl(smpl_data, frame_to_extract, output_filename)


if __name__ == "__main__":
    main()
