import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def read_mot_file(file_path):
    """Read a .mot file and return a pandas DataFrame."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find the header row (starts with "endheader")
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("endheader"):
            header_idx = i + 1
            break

    # Get column names from the line after endheader
    column_names = lines[header_idx].strip().split()

    # Parse the data rows
    data_rows = []
    for line in lines[header_idx + 1 :]:
        if line.strip():  # Skip empty lines
            data_rows.append([float(x) for x in line.strip().split()])

    # Create DataFrame
    return pd.DataFrame(data_rows, columns=column_names)


def main():
    # File paths
    csv_file = "/home/selim/Downloads/STS1_5_sync_pred_grfs.csv"
    mot_file = "/home/selim/opencap-mono/LabValidation_withVideos1/subject8/ForceData/STS1_forces.mot"

    # Read the files
    print(f"Reading CSV file: {csv_file}")
    csv_data = pd.read_csv(csv_file)

    print(f"Reading MOT file: {mot_file}")
    mot_data = read_mot_file(mot_file)

    # Print basic info about the data
    print("\nCSV data shape:", csv_data.shape)
    print("CSV columns:", csv_data.columns.tolist())
    print("\nMOT data shape:", mot_data.shape)
    print("MOT columns:", mot_data.columns.tolist())

    # Plot the data
    plot_forces(csv_data, mot_data)


def plot_forces(csv_data, mot_data):
    """Plot force data from both sources for comparison."""
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle("Force Data Comparison", fontsize=16)

    # Get time columns based on actual data
    csv_time_col = "time (sec)" if "time (sec)" in csv_data.columns else "Unnamed: 0"
    mot_time_col = "time"

    # Body weight in kg for scaling
    body_weight = 60  # kg
    gravity = 9.81  # m/s²

    # Mapping of force components to plot based on actual column names
    force_components = {
        "Ground Reaction Forces X": {
            "csv": ["GRF_x_right (BW)", "GRF_x_left (BW)"],
            "mot": ["R_ground_force_vx", "L_ground_force_vx"],
        },
        "Ground Reaction Forces Y": {
            "csv": ["GRF_y_right (BW)", "GRF_y_left (BW)"],
            "mot": ["R_ground_force_vy", "L_ground_force_vy"],
        },
        "Ground Reaction Forces Z": {
            "csv": ["GRF_z_right (BW)", "GRF_z_left (BW)"],
            "mot": ["R_ground_force_vz", "L_ground_force_vz"],
        },
        "Ground Reaction Moments X": {
            "csv": [],  # CSV doesn't seem to have torque data
            "mot": ["R_ground_torque_x", "L_ground_torque_x"],
        },
        "Ground Reaction Moments Y": {
            "csv": [],  # CSV doesn't seem to have torque data
            "mot": ["R_ground_torque_y", "L_ground_torque_y"],
        },
        "Ground Reaction Moments Z": {
            "csv": [],  # CSV doesn't seem to have torque data
            "mot": ["R_ground_torque_z", "L_ground_torque_z"],
        },
    }

    # Plot each component
    for i, (title, columns) in enumerate(force_components.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        # Plot CSV data - scale by body weight
        for col_name in columns["csv"]:
            if col_name in csv_data.columns:
                # Convert from BW to Newtons (BW * body_weight * gravity)
                scaled_data = csv_data[col_name] * body_weight * gravity
                ax.plot(
                    csv_data[csv_time_col],
                    scaled_data,
                    label=f"CSV {col_name} (scaled)",
                )

        # Plot MOT data
        for col_name in columns["mot"]:
            if col_name in mot_data.columns:
                ax.plot(
                    mot_data[mot_time_col],
                    mot_data[col_name],
                    "--",
                    label=f"MOT {col_name}",
                )

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (N) / Moment (Nm)")
        if any(col_name in csv_data.columns for col_name in columns["csv"]) or any(
            col_name in mot_data.columns for col_name in columns["mot"]
        ):
            ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig("force_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
