import pickle
import argparse
import os
import open3d as o3d
import numpy as np
import time
import sys
from typing import Dict, Any

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

# Defines the connections between keypoints to form a skele
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


def convert_to_openpose_25(keypoints_wholebody: np.ndarray) -> np.ndarray:
    """
    Converts 133 whole-body keypoints to the 25-point OpenPose format.
    The input is (frames, 133, 2) and output is (frames, 25, 2).
    """
    num_frames = keypoints_wholebody.shape[0]
    openpose_keypoints = np.full((num_frames, 25, 2), np.nan)

    # Direct mapping for 23 keypoints from whole-body to OpenPose
    # OpenPose_idx: WholeBody_idx
    mapping = {
        0: 0,  # Nose
        2: 6,  # RShoulder
        3: 8,  # RElbow
        4: 10,  # RWrist
        5: 5,  # LShoulder
        6: 7,  # LElbow
        7: 9,  # LWrist
        9: 12,  # RHip
        10: 14,  # RKnee
        11: 16,  # RAnkle
        12: 11,  # LHip
        13: 13,  # LKnee
        14: 15,  # LAnkle
        15: 1,  # REye
        16: 2,  # LEye
        17: 3,  # REar
        18: 4,  # LEar
        19: 20,  # LBigToe
        20: 21,  # LSmallToe
        21: 22,  # LHeel
        22: 17,  # RBigToe
        23: 18,  # RSmallToe
        24: 19,  # RHeel
    }
    for op_idx, wb_idx in mapping.items():
        openpose_keypoints[:, op_idx, :] = keypoints_wholebody[:, wb_idx, :]

    # Calculated keypoints: Neck and MidHip
    # Neck (1) is the midpoint of LShoulder (5) and RShoulder (2)
    openpose_keypoints[:, 1, :] = np.mean(openpose_keypoints[:, [5, 2], :], axis=1)
    # MidHip (8) is the midpoint of LHip (12) and RHip (9)
    openpose_keypoints[:, 8, :] = np.mean(openpose_keypoints[:, [12, 9], :], axis=1)

    return openpose_keypoints


def load_intrinsics(calib_path: str) -> Dict[str, Any]:
    """Loads camera intrinsics from a file."""
    if calib_path.endswith(".pickle"):
        with open(calib_path, "rb") as file:
            intrinsics_dict = pickle.load(file)
        intrinsics = {
            "fx": intrinsics_dict["intrinsicMat"][0, 0],
            "fy": intrinsics_dict["intrinsicMat"][1, 1],
            "cx": intrinsics_dict["intrinsicMat"][0, 2],
            "cy": intrinsics_dict["intrinsicMat"][1, 2],
        }
    else:
        calib = np.loadtxt(calib_path, delimiter=" ")
        fx, fy, cx, cy = calib[:4]
        intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return intrinsics


def create_3d_keypoints(
    keypoints_2d: np.ndarray, depth_maps: np.ndarray, intrinsics: Dict[str, Any]
) -> np.ndarray:
    """
    Projects 2D keypoints to 3D space using depth maps and camera intrinsics.
    """
    num_frames, num_keypoints, _ = keypoints_2d.shape
    keypoints_3d = np.full((num_frames, num_keypoints, 3), np.nan)

    fx, fy, cx, cy = (
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )
    depth_height, depth_width = depth_maps.shape[1], depth_maps.shape[2]

    for f in range(num_frames):
        for k in range(num_keypoints):
            u, v = keypoints_2d[f, k]

            if np.isnan(u) or np.isnan(v):
                continue

            # Round and clamp coordinates to be within depth map bounds
            u_idx, v_idx = int(round(u)), int(round(v))
            if not (0 <= u_idx < depth_width and 0 <= v_idx < depth_height):
                continue

            z = depth_maps[f, v_idx, u_idx]

            # Assuming depth is in meters. If in mm, z /= 1000.0
            # Also, filter out zero or invalid depth values
            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            keypoints_3d[f, k, :] = [x, y, z]

    return keypoints_3d


# Global state for animation control
animation_state = {"is_paused": False}


def toggle_pause(vis):
    """Callback function to toggle the pause state of the animation."""
    animation_state["is_paused"] = not animation_state["is_paused"]
    return False


def visualize_3d_skeleton(keypoints_data: np.ndarray, fps: int = 30):
    """
    Creates an interactive 3D animation of the skeleton using Open3D.
    """
    if keypoints_data.shape[0] == 0:
        print("Error: The keypoints data is empty.")
        return

    num_frames = keypoints_data.shape[0]
    frame_duration = 1.0 / fps

    points = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Skeleton Animation from 2D+Depth")
    vis.register_key_callback(ord(" "), toggle_pause)

    # Add geometries once
    is_geom_added = False

    frame_idx = 0
    last_update_time = time.time()

    while True:
        current_time = time.time()
        if not animation_state["is_paused"] and (
            current_time - last_update_time >= frame_duration
        ):
            current_keypoints = keypoints_data[frame_idx, :, :]

            # Filter out NaN points for visualization
            valid_kpts_mask = ~np.isnan(current_keypoints).any(axis=1)
            valid_kpts = current_keypoints[valid_kpts_mask]

            if valid_kpts.shape[0] > 0:
                points.points = o3d.utility.Vector3dVector(valid_kpts)
                points.paint_uniform_color([1, 0, 0])

                # Remap skeleton connections for valid points
                point_map = {
                    old_idx: new_idx
                    for new_idx, old_idx in enumerate(np.where(valid_kpts_mask)[0])
                }
                valid_lines = []
                for i, j in SKELETON_CONNECTIONS:
                    if i in point_map and j in point_map:
                        valid_lines.append([point_map[i], point_map[j]])

                lines.points = o3d.utility.Vector3dVector(valid_kpts)
                lines.lines = o3d.utility.Vector2iVector(valid_lines)
                lines.paint_uniform_color([0, 0, 1])

                if not is_geom_added:
                    vis.add_geometry(points)
                    vis.add_geometry(lines)
                    is_geom_added = True
                else:
                    vis.update_geometry(points)
                    vis.update_geometry(lines)

            status_text = (
                f"\rFrame: {frame_idx+1}/{num_frames} | "
                f"{'PAUSED' if animation_state['is_paused'] else 'PLAYING'}. Press SPACE to toggle."
            )
            sys.stdout.write(status_text)
            sys.stdout.flush()

            frame_idx = (frame_idx + 1) % num_frames
            last_update_time = current_time

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize 3D skeleton from 2D keypoints and depth."
    )
    parser.add_argument(
        "keypoints_2d_path", type=str, help="Path to the 2D keypoints .npy file."
    )
    parser.add_argument(
        "depth_path", type=str, help="Path to the depth data file (.npz or .npy)."
    )
    parser.add_argument(
        "intrinsics_path", type=str, help="Path to the camera intrinsics file."
    )
    parser.add_argument(
        "--depth_key",
        type=str,
        default="depth",
        help="Key for depth maps in the .npz file (if applicable).",
    )
    parser.add_argument(
        "--depth_unit",
        type=str,
        default="m",
        choices=["m", "mm"],
        help="Unit of the depth values (m or mm).",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the animation."
    )
    args = parser.parse_args()

    print("Loading 2D keypoints...")
    keypoints_2d_raw = np.load(args.keypoints_2d_path)

    if keypoints_2d_raw.shape[1] == 133:
        print("Converting 133 whole-body keypoints to 25 OpenPose keypoints...")
        keypoints_2d = convert_to_openpose_25(keypoints_2d_raw)
    elif keypoints_2d_raw.shape[1] == 25:
        keypoints_2d = keypoints_2d_raw
    else:
        print(
            f"Error: Unexpected number of keypoints ({keypoints_2d_raw.shape[1]}). Expected 25 or 133."
        )
        sys.exit(1)

    print("Loading depth maps...")
    if args.depth_path.endswith(".npy"):
        depth_maps = np.load(args.depth_path)
    elif args.depth_path.endswith(".npz"):
        depth_data = np.load(args.depth_path)
        if args.depth_key not in depth_data:
            print(f"Error: Key '{args.depth_key}' not found in '{args.depth_path}'.")
            print(f"Available keys: {list(depth_data.keys())}")
            sys.exit(1)
        depth_maps = depth_data[args.depth_key]
    else:
        print(
            f"Error: Unsupported depth file format for '{args.depth_path}'. Please use .npy or .npz."
        )
        sys.exit(1)

    if args.depth_unit == "mm":
        print("Converting depth from millimeters to meters...")
        depth_maps = depth_maps / 1000.0

    print("Loading camera intrinsics...")
    intrinsics = load_intrinsics(args.intrinsics_path)

    print("Creating 3D keypoints from 2D keypoints and depth...")
    keypoints_3d = create_3d_keypoints(keypoints_2d, depth_maps, intrinsics)

    print("Starting visualization...")
    visualize_3d_skeleton(keypoints_3d, args.fps)

    print("\nVisualization finished.")
