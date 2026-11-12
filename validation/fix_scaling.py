"""
script to load marker trc file and commpute the distance between two markers
"""

from loguru import logger
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils_trc import TRCFile


def _load_trc(trc_path):
    """
    Load marker data from TRC file

    Returns:
    --------
    marker_data: numpy.ndarray
        Marker positions (n_frames, n_markers, 3)
    marker_names: list
        List of marker names
    frame_rate: float
        Frame rate of the marker data
    """
    logger.info(f"Loading marker data from {trc_path}")

    # Load TRC file
    trc_file = TRCFile(trc_path)

    # Extract marker names
    marker_names = trc_file.marker_names

    # Extract frame rate
    frame_rate = trc_file.data_rate

    # Extract marker data
    n_frames = trc_file.num_frames
    n_markers = trc_file.num_markers

    # Initialize marker data array
    marker_data = np.zeros((n_frames, n_markers, 3))

    # Fill marker data array
    for i, marker_name in enumerate(marker_names):
        marker_data[:, i, 0] = trc_file.data[marker_name + "_tx"]
        marker_data[:, i, 1] = trc_file.data[marker_name + "_ty"]
        marker_data[:, i, 2] = trc_file.data[marker_name + "_tz"]

    return marker_data, marker_names, frame_rate


if __name__ == "__main__":
    dataset = "LabValidation_withVideos1"

    # subjects = [f'subject{i}' for i in range(2, 12)]
    subjects = ["subject4"]

    # cases = ['case_001_walking', 'case_001_squats', 'case_001_STS']

    case = "case_fixed_10"
    session = "Session0"

    distances_neutral_right = []
    distances_sync_right = []
    distances_neutral_left = []
    distances_sync_left = []

    for subject in subjects:
        try:
            # Load sync TRC file
            sync_trc = f"output/{case}/{subject}/{session}/Cam3/squats1/MarkerData/squats1_5/squats1_5_sync.trc"
            marker_data_sync, marker_names_sync, frame_rate_sync = _load_trc(sync_trc)
            print("Sync marker names: ", marker_names_sync)

            # Define marker pairs for both sides (sync data)
            # Right side markers
            marker1_sync_right = "C7"
            marker2_sync_right = "r_elbow"

            # Left side markers
            marker1_sync_left = "C7"
            marker2_sync_left = "l_elbow"

            # Get marker indices for right side
            marker1_idx_sync_right = marker_names_sync.index(marker1_sync_right)
            marker2_idx_sync_right = marker_names_sync.index(marker2_sync_right)

            # Get marker indices for left side
            marker1_idx_sync_left = marker_names_sync.index(marker1_sync_left)
            marker2_idx_sync_left = marker_names_sync.index(marker2_sync_left)

            # Calculate distances for right side
            distance_sync_right = (
                np.linalg.norm(
                    marker_data_sync[:, marker1_idx_sync_right, :]
                    - marker_data_sync[:, marker2_idx_sync_right, :],
                    axis=1,
                )
                / 1000
            )
            mean_distance_sync_right = np.mean(distance_sync_right)
            distances_sync_right.append(mean_distance_sync_right)
            print(
                f"SYNC RIGHT: Mean distance between {marker1_sync_right} and {marker2_sync_right}: {mean_distance_sync_right:.2f} m"
            )

            # Calculate distances for left side
            distance_sync_left = (
                np.linalg.norm(
                    marker_data_sync[:, marker1_idx_sync_left, :]
                    - marker_data_sync[:, marker2_idx_sync_left, :],
                    axis=1,
                )
                / 1000
            )
            mean_distance_sync_left = np.mean(distance_sync_left)
            distances_sync_left.append(mean_distance_sync_left)
            print(
                f"SYNC LEFT: Mean distance between {marker1_sync_left} and {marker2_sync_left}: {mean_distance_sync_left:.2f} m"
            )

            case_name = "squats1_5"

            # Load neutral TRC file
            neutral_trc = f"output/{case}/{subject}/{session}/Cam3/squats1/OpenSim/Model/{case_name}/neutral.trc"
            marker_data_neutral, marker_names_neutral, frame_rate_neutral = _load_trc(
                neutral_trc
            )
            print("Neutral marker names: ", marker_names_neutral)

            # Define marker pairs for both sides (neutral data)
            # Right side markers
            marker1_neutral_right = "C7"
            marker2_neutral_right = "r_elbow"

            # Left side markers
            marker1_neutral_left = "C7"
            marker2_neutral_left = "l_elbow"

            # Get marker indices for right side
            marker1_idx_neutral_right = marker_names_neutral.index(
                marker1_neutral_right
            )
            marker2_idx_neutral_right = marker_names_neutral.index(
                marker2_neutral_right
            )

            # Get marker indices for left side
            marker1_idx_neutral_left = marker_names_neutral.index(marker1_neutral_left)
            marker2_idx_neutral_left = marker_names_neutral.index(marker2_neutral_left)

            # Calculate distances for right side
            distance_neutral_right = np.linalg.norm(
                marker_data_neutral[:, marker1_idx_neutral_right, :]
                - marker_data_neutral[:, marker2_idx_neutral_right, :],
                axis=1,
            )
            mean_distance_neutral_right = np.mean(distance_neutral_right)
            distances_neutral_right.append(mean_distance_neutral_right)
            print(
                f"NEUTRAL RIGHT: Mean distance between {marker1_neutral_right} and {marker2_neutral_right}: {mean_distance_neutral_right:.2f} m"
            )

            # Calculate distances for left side
            distance_neutral_left = np.linalg.norm(
                marker_data_neutral[:, marker1_idx_neutral_left, :]
                - marker_data_neutral[:, marker2_idx_neutral_left, :],
                axis=1,
            )
            mean_distance_neutral_left = np.mean(distance_neutral_left)
            distances_neutral_left.append(mean_distance_neutral_left)
            print(
                f"NEUTRAL LEFT: Mean distance between {marker1_neutral_left} and {marker2_neutral_left}: {mean_distance_neutral_left:.2f} m"
            )

            # Create time vectors for plotting
            time_sync = np.arange(len(distance_sync_right)) / frame_rate_sync
            time_neutral = np.arange(len(distance_neutral_right)) / frame_rate_neutral

            # Print time range information
            print(
                f"SYNC time range: {time_sync.min():.2f} to {time_sync.max():.2f} seconds (duration: {time_sync.max() - time_sync.min():.2f} s)"
            )
            print(
                f"NEUTRAL time range: {time_neutral.min():.2f} to {time_neutral.max():.2f} seconds (duration: {time_neutral.max() - time_neutral.min():.2f} s)"
            )
            print(
                f"SYNC frame rate: {frame_rate_sync:.2f} Hz, frames: {len(distance_sync_right)}"
            )
            print(
                f"NEUTRAL frame rate: {frame_rate_neutral:.2f} Hz, frames: {len(distance_neutral_right)}"
            )

            # Create the SYNC plot
            plt.figure(figsize=(12, 8))

            # Plot SYNC distances for each frame
            plt.plot(
                time_sync,
                distance_sync_right,
                "b-",
                label=f"RIGHT: {marker1_sync_right} - {marker2_sync_right}",
                linewidth=2,
            )
            plt.plot(
                time_sync,
                distance_sync_left,
                "r-",
                label=f"LEFT: {marker1_sync_left} - {marker2_sync_left}",
                linewidth=2,
            )

            # Plot SYNC average lines
            plt.axhline(
                y=mean_distance_sync_right,
                color="b",
                linestyle="--",
                alpha=0.7,
                label=f"RIGHT avg: {mean_distance_sync_right:.2f} m",
            )
            plt.axhline(
                y=mean_distance_sync_left,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"LEFT avg: {mean_distance_sync_left:.2f} m",
            )

            plt.xlabel("Time (seconds)")
            plt.ylabel("Distance (m)")
            plt.title(
                f"SYNC: {marker1_sync_right} - {marker2_sync_right} Distance Over Time - {subject}"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the SYNC plot
            os.makedirs("plots", exist_ok=True)
            plt.savefig(
                f"plots/sync_distance_comparison_{subject}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

            # Create the NEUTRAL plot
            plt.figure(figsize=(12, 8))

            # Plot NEUTRAL distances for each frame
            plt.plot(
                time_neutral,
                distance_neutral_right,
                "b-",
                label=f"RIGHT: {marker1_neutral_right} - {marker2_neutral_right}",
                linewidth=2,
            )
            plt.plot(
                time_neutral,
                distance_neutral_left,
                "r-",
                label=f"LEFT: {marker1_neutral_left} - {marker2_neutral_left}",
                linewidth=2,
            )

            # Plot NEUTRAL average lines
            plt.axhline(
                y=mean_distance_neutral_right,
                color="b",
                linestyle="--",
                alpha=0.7,
                label=f"RIGHT avg: {mean_distance_neutral_right:.2f} m",
            )
            plt.axhline(
                y=mean_distance_neutral_left,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"LEFT avg: {mean_distance_neutral_left:.2f} m",
            )

            plt.xlabel("Time (seconds)")
            plt.ylabel("Distance (m)")
            plt.title(f"NEUTRAL: Shoulder-Elbow Distance Over Time - {subject}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the NEUTRAL plot
            plt.savefig(
                f"plots/neutral_distance_comparison_{subject}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

            print(f"Plots saved as:")
            print(f"  - plots/sync_distance_comparison_{subject}.png")
            print(f"  - plots/neutral_distance_comparison_{subject}.png")

        except Exception as e:
            print(f"Error for subject {subject}: {e}")
            continue

    print("\n")
    print(f"Number of subjects processed: {len(distances_sync_right)}")

    if distances_sync_right and distances_neutral_right:
        print("\n=== SYNC RIGHT SIDE STATISTICS ===")
        print(f"SYNC RIGHT distances: {distances_sync_right}")
        print(f"NEUTRAL RIGHT distances: {distances_neutral_right}")
        print(f"Mean SYNC RIGHT distance: {np.mean(distances_sync_right):.2f} m")
        print(f"Mean NEUTRAL RIGHT distance: {np.mean(distances_neutral_right):.2f} m")
        print(f"Median SYNC RIGHT distance: {np.median(distances_sync_right):.2f} m")
        print(
            f"Median NEUTRAL RIGHT distance: {np.median(distances_neutral_right):.2f} m"
        )
        print(f"Std SYNC RIGHT distance: {np.std(distances_sync_right):.2f} m")
        print(f"Std NEUTRAL RIGHT distance: {np.std(distances_neutral_right):.2f} m")

    if distances_sync_left and distances_neutral_left:
        print("\n=== SYNC LEFT SIDE STATISTICS ===")
        print(f"SYNC LEFT distances: {distances_sync_left}")
        print(f"NEUTRAL LEFT distances: {distances_neutral_left}")
        print(f"Mean SYNC LEFT distance: {np.mean(distances_sync_left):.2f} m")
        print(f"Mean NEUTRAL LEFT distance: {np.mean(distances_neutral_left):.2f} m")
        print(f"Median SYNC LEFT distance: {np.median(distances_sync_left):.2f} m")
        print(
            f"Median NEUTRAL LEFT distance: {np.median(distances_neutral_left):.2f} m"
        )
        print(f"Std SYNC LEFT distance: {np.std(distances_sync_left):.2f} m")
        print(f"Std NEUTRAL LEFT distance: {np.std(distances_neutral_left):.2f} m")

    # print(distances_mocap)
    # # get the mean distance
    # mean_distance = np.round(np.mean(distances_mocap), 4)
    # print(f"MOCAP: Mean distance between {marker2_mocap} and {marker2_mocap}: {mean_distance} mm")
    # print('\n')

    # # same for mono
    # print(distances_mono)
    # # get the mean distance
    # mean_distance = np.round(np.mean(distances_mono), 4)
    # print(f"MONO: Mean distance between {marker2_mono} and {marker2_mono}: {mean_distance} mm")
