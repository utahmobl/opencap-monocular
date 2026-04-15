import pickle
import argparse
import os
import open3d as o3d
import numpy as np
import time
import sys

# A list of keypoint names for labeling. Based on OpenPose body_25.
KEYPOINT_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]

# Defines the connections between keypoints to form a skeleton
SKELETON_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (11, 22),
    (11, 23),
    (11, 24),
    (14, 19),
    (14, 20),
    (14, 21),
]

# Global state for animation control
animation_state = {"is_paused": False}


def toggle_pause(vis):
    """Callback function to toggle the pause state of the animation."""
    animation_state["is_paused"] = not animation_state["is_paused"]
    return False


def visualize_3d_skeleton(file_path, fps=30):
    """
    Loads 3D keypoint data and creates an interactive 3D animation of the
    skeleton using Open3D.

    Args:
        file_path (str): The path to the keypoints_3d_cam.pkl file.
        fps (int): The frame rate for the animation.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, "rb") as f:
        keypoints_data = pickle.load(f)

    if keypoints_data.shape[0] == 0:
        print("Error: The keypoints data is empty.")
        return

    num_frames = keypoints_data.shape[0]
    frame_duration = 1.0 / fps

    # --- Initialize Geometries with the first frame ---
    first_frame_keypoints = keypoints_data[0, :, :]

    # Create PointCloud for joints and LineSet for bones
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(first_frame_keypoints)
    points.paint_uniform_color([1, 0, 0])  # Red joints

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(first_frame_keypoints)
    lines.lines = o3d.utility.Vector2iVector(SKELETON_CONNECTIONS)
    lines.paint_uniform_color([0, 0, 1])  # Blue bones

    # --- Setup Open3D Visualization ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Skeleton Animation")

    # Register the key callback for the spacebar to toggle pause
    vis.register_key_callback(ord(" "), toggle_pause)

    # Add geometries to the visualizer
    vis.add_geometry(points)
    vis.add_geometry(lines)

    # --- Animation Loop ---
    frame_idx = 0
    last_update_time = time.time()

    keep_running = True
    while keep_running:
        current_time = time.time()
        if not animation_state["is_paused"] and (
            current_time - last_update_time >= frame_duration
        ):
            # Get keypoints for the current frame
            current_keypoints = keypoints_data[frame_idx, :, :]

            # Update PointCloud
            points.points = o3d.utility.Vector3dVector(current_keypoints)
            points.paint_uniform_color([1, 0, 0])  # Red joints

            # Update LineSet
            lines.points = o3d.utility.Vector3dVector(current_keypoints)
            lines.lines = o3d.utility.Vector2iVector(SKELETON_CONNECTIONS)
            lines.paint_uniform_color([0, 0, 1])  # Blue bones

            # Update the geometries in the visualizer
            vis.update_geometry(points)
            vis.update_geometry(lines)

            # Get coordinates for logging
            neck_coords = current_keypoints[1]  # Index 1 is Neck
            midhip_coords = current_keypoints[8]  # Index 8 is MidHip
            rhip_coords = current_keypoints[9]  # Index 9 is RHip
            lhip_coords = current_keypoints[12]  # Index 12 is LHip

            # Print status to the console
            status_text = (
                f"\rFrame: {frame_idx+1}/{num_frames} | "
                f"Depth (m) -> Neck: {neck_coords[2]:.2f}, Mid-Hip: {midhip_coords[2]:.2f}, R-Hip: {rhip_coords[2]:.2f}, L-Hip: {lhip_coords[2]:.2f} | "
                f"{'PAUSED' if animation_state['is_paused'] else 'PLAYING'}. Press SPACE to toggle."
            )
            sys.stdout.write(status_text)
            sys.stdout.flush()

            # Move to the next frame
            frame_idx = (frame_idx + 1) % num_frames
            last_update_time = current_time

        # Process events and render the frame
        if not vis.poll_events():
            keep_running = False
        vis.update_renderer()

    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a 3D animated skeleton from a .pkl file using Open3D."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the keypoints_3d_cam.pkl file."
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the animation."
    )
    args = parser.parse_args()
    visualize_3d_skeleton(args.file_path, args.fps)
