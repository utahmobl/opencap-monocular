import smplx
import torch
import trimesh
import os
import pandas as pd
import numpy as np
import pickle  # Add this import

# === CONFIGURATION ===
model_folder = "WHAM/dataset/body_models/smplx"  # Folder containing SMPL-X model files (e.g., SMPLX_MALE.npz)
gender = "neutral"  # 'male', 'female', or 'neutral'
num_betas = 10  # Shape parameters
num_expression_coeffs = 10  # For facial expression (optional)
output_path = "smplx_output.obj"  # Path to save the .obj file

# === NEW: Option to load from pickle file ===
use_pickle_params = True  # Set to True to load from pickle, False for default pose
script_dir = os.path.dirname(os.path.abspath(__file__))
# The pickle file is in the same directory as this script
pickle_file_path = os.path.join(
    script_dir, "optimized_smpl_params_frame_10.pkl"
)  # Path to pickle file


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


# === LOAD SMPL-X MODEL ===
model = smplx.SMPLX(
    model_folder,
    gender=gender,
    batch_size=1,
    use_pca=False,
    num_betas=num_betas,
    num_expression_coeffs=num_expression_coeffs,
)

# === LOAD PARAMETERS ===
if use_pickle_params and os.path.exists(pickle_file_path):
    print(f"Loading parameters from {pickle_file_path}")
    smpl_params = load_smpl_params(pickle_file_path)

    if smpl_params is not None and isinstance(smpl_params, dict):
        # Extract betas (shape parameters)
        if "betas" in smpl_params:
            loaded_betas = torch.tensor(smpl_params["betas"], dtype=torch.float32)
            if loaded_betas.dim() == 1:
                loaded_betas = loaded_betas.unsqueeze(0)
            betas = loaded_betas[:, :num_betas]  # Use only the required number of betas
            print(f"Loaded betas from pickle file: {betas}")
        else:
            print("Warning: 'betas' not found in pickle file, using default")
            betas = torch.zeros([1, num_betas])

        # Extract body pose
        body_pose_loaded = None
        if "poses" in smpl_params:
            full_poses = torch.tensor(smpl_params["poses"], dtype=torch.float32)
            if full_poses.dim() == 1:
                full_poses = full_poses.unsqueeze(0)
            # Body pose is typically poses[3:] (excluding global rotation)
            if full_poses.shape[-1] > 3:
                body_pose_loaded = full_poses[
                    :, 3:72
                ]  # SMPL-X body pose is 21*3=63 params
                global_orient = full_poses[:, :3]  # First 3 are global orientation
            else:
                body_pose_loaded = full_poses
                global_orient = torch.zeros([1, 3])
        elif "body_pose" in smpl_params:
            body_pose_loaded = torch.tensor(
                smpl_params["body_pose"], dtype=torch.float32
            )
            if body_pose_loaded.dim() == 1:
                body_pose_loaded = body_pose_loaded.unsqueeze(0)

            # Also load global_orient if it exists
            if "global_orient" in smpl_params:
                global_orient = torch.tensor(
                    smpl_params["global_orient"], dtype=torch.float32
                )
                if global_orient.dim() == 1:
                    global_orient = global_orient.unsqueeze(0)
            else:
                global_orient = torch.zeros([1, 3])

        if body_pose_loaded is not None:
            # Ensure correct size for SMPL-X (21 joints * 3 = 63 parameters)
            if body_pose_loaded.shape[-1] < 63:
                body_pose = torch.zeros([1, 63])
                body_pose[:, : body_pose_loaded.shape[-1]] = body_pose_loaded
            else:
                body_pose = body_pose_loaded[:, :63]
        else:
            print("Warning: body pose not found in pickle file, using default")
            body_pose = torch.zeros([1, 21 * 3])
            global_orient = torch.zeros([1, 3])

        # Extract translation
        if "transl" in smpl_params:
            transl = torch.tensor(smpl_params["transl"], dtype=torch.float32)
            if transl.dim() == 1:
                transl = transl.unsqueeze(0)
        else:
            print("Warning: 'transl' not found in pickle file, using default")
            transl = torch.zeros([1, 3])

        print(f"Loaded betas shape: {betas.shape}")
        print(f"Loaded body_pose shape: {body_pose.shape}")
        print(f"Loaded global_orient shape: {global_orient.shape}")
        print(f"Loaded transl shape: {transl.shape}")
    else:
        print("Failed to load pickle file, using default parameters")
        betas = torch.zeros([1, num_betas])
        body_pose = torch.zeros([1, 21 * 3])
        global_orient = torch.zeros([1, 3])
        transl = torch.zeros([1, 3])
else:
    print("Using default parameters")
    # === NEUTRAL POSE AND SHAPE ===
    betas = torch.zeros([1, num_betas])
    body_pose = torch.zeros([1, 21 * 3])  # 21 joints * 3 (axis-angle)
    global_orient = torch.zeros([1, 3])  # No rotation
    transl = torch.zeros([1, 3])

# === REMAINING PARAMETERS (always default for now) ===
left_hand_pose = torch.zeros([1, 15 * 3])  # Optional hand pose
right_hand_pose = torch.zeros([1, 15 * 3])
jaw_pose = torch.zeros([1, 3])
expression = torch.zeros([1, num_expression_coeffs])

# === FLEX ELBOWS (only if using default parameters) ===
if not use_pickle_params or not os.path.exists(pickle_file_path):
    # In SMPL-X, elbow joints are at indices:
    # Left elbow: joint 18 (body_pose indices 54-56)
    # Right elbow: joint 19 (body_pose indices 57-59)
    flexion_angle = np.pi / 1.5  # 60 degrees flexion
    flexion_angle = 0

    # Flex both elbows (rotation around Y-axis for natural flexion)
    body_pose[0, 17 * 3 + 1] = -flexion_angle  # Left elbow flexion (index 55)
    body_pose[0, 18 * 3 + 1] = flexion_angle  # Right elbow flexion (index 58)

    print(f"Applied elbow flexion: {np.degrees(flexion_angle):.1f} degrees")

output = model(
    betas=betas,
    body_pose=body_pose,
    global_orient=global_orient,
    transl=transl,
    left_hand_pose=left_hand_pose,
    right_hand_pose=right_hand_pose,
    jaw_pose=jaw_pose,
    expression=expression,
    return_verts=True,
)

# === GET VERTICES AND FACES ===
vertices = output.vertices.detach().cpu().numpy().squeeze()
faces = model.faces  # numpy array of shape (N, 3)

# === EXPORT MESH AS OBJ ===
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
mesh.export(output_path)

print(f"SMPL-X mesh exported to: {os.path.abspath(output_path)}")


from vedo import Mesh, Plotter, Point
import trimesh

# === Load your .obj file ===
mesh_path = "smplx_output.obj"  # Replace with your mesh path
tmesh = trimesh.load(mesh_path)
vmesh = Mesh([tmesh.vertices, tmesh.faces])
vmesh.c("lightblue").point_size(5)
vmesh.alpha(0.6)  # Make mesh slightly transparent to see vertices
vmesh.lw(0.5)  # Draw mesh edges (faces) with thin lines
vmesh.wireframe(True)  # Show mesh faces as wireframe

# === Create plotter ===
plt = Plotter(title="Click a vertex to see its index", axes=1)

# === Callback function for mouse click ===
highlighted = []


def on_click(evt):
    if not evt.actor:
        return

    vid = evt.actor.closest_point(evt.picked3d, return_point_id=True)
    vpos = evt.actor.points[vid]

    print(f"Clicked vertex index: {vid}, position: {vpos}")

    # Do NOT remove previously highlighted points
    point = Point(vpos, r=12, c="red")
    highlighted.append(point)
    plt.add(point)


def highlight_vertex(vid):
    """Highlight a vertex by its index."""
    points = vmesh.points
    if vid < 0 or vid >= len(points):
        print(f"Vertex index {vid} is out of range.")
        return
    vpos = points[vid]
    for p in highlighted:
        p.remove()
    highlighted.clear()
    point = Point(vpos, r=12, c="green")
    highlighted.append(point)
    plt.add(point)
    plt.render()
    print(f"Highlighted vertex index: {vid}, position: {vpos}")


# Attach the click callback
plt.add_callback("mouse click", on_click)

# Highlight vertex 5895 *before* showing the window
# highlight_vertex(5895)

# === Highlight keypoints from CSV ===
csv_path = "utils/data/vertices_keypoints_corr.csv"  # Adjust path if needed

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, header=0)  # Use header=0 to skip the header row
    # Do NOT subtract 1 if indices are already 0-based
    keypoint_indices = (
        pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().astype(int).tolist()
    )
    points = vmesh.points
    for idx in keypoint_indices:
        if 0 <= idx < len(points):
            point = Point(points[idx], r=14, c="orange")
            highlighted.append(point)
            plt.add(point)
    print(f"Highlighted {len(keypoint_indices)} keypoints from {csv_path}")
else:
    print(f"CSV file not found: {csv_path}")

# Show the mesh and interactive window
plt.show(vmesh, "Click a vertex to see index", interactive=True)
