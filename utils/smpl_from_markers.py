#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create SMPL models from marker data and motion files
"""

import os
import torch
import numpy as np
from loguru import logger
import pickle
from utils.opensim.utils_opensim import smpl_to_smplx_verts, smplx_to_markers
from utils.utils_optim import pred_smpl
from slahmr.slahmr.util.loaders import load_smpl_body_model
from utils.opensim.defaults import defaults
from utils.utils_trc import TRCFile
import pandas as pd


class MarkerToSMPL:
    """
    Class to create SMPL models from marker data (.trc) and motion files (.mot)
    """

    def __init__(self, trc_path, mot_path, output_dir, gender="neutral", device=None):
        """
        Initialize the MarkerToSMPL class

        Parameters:
        -----------
        trc_path: str
            Path to the marker data file (.trc)
        mot_path: str
            Path to the motion file (.mot)
        output_dir: str
            Directory to save the output files
        gender: str
            Gender for the SMPL model ('male', 'female', or 'neutral')
        device: torch.device
            Device to run the model on (CPU or GPU)
        """
        self.trc_path = trc_path
        self.mot_path = mot_path
        self.output_dir = output_dir
        self.gender = gender

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load SMPL model
        self.smpl_model_path = defaults()["SMPL_NEUTRAL_PATH"]
        self.smpl_model, _ = load_smpl_body_model(
            path=self.smpl_model_path,
            batch_size=1,
            num_betas=10,
            model_type="smpl",
            use_vtx_selector=True,
            device=self.device,
            fit_gender=gender,
            npz_hack=False,
        )

        # Load marker data
        self.marker_data, self.marker_names, self.frame_rate = self._load_trc()

        # Load motion data
        self.motion_data = self._load_mot()

    def _load_trc(self):
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
        logger.info(f"Loading marker data from {self.trc_path}")

        # Load TRC file
        trc_file = TRCFile(self.trc_path)

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

    def _load_mot(self):
        """
        Load motion data from MOT file

        Returns:
        --------
        motion_data: pandas.DataFrame
            Motion data
        """
        logger.info(f"Loading motion data from {self.mot_path}")
        motion_data = pd.read_csv(self.mot_path, skiprows=10, delimiter="\t")
        return motion_data

    def estimate_smpl_parameters(self, beta=None):
        """
        Estimate SMPL parameters from marker data and motion file

        Parameters:
        -----------
        beta: torch.Tensor
            SMPL shape parameters (if None, will use default)

        Returns:
        --------
        smpl_params: dict
            Dictionary containing SMPL parameters
        """
        logger.info("Estimating SMPL parameters from marker and motion data")

        # Extract motion parameters from MOT file
        # This is a simplified approach - you'll need to adapt this to your specific MOT file format
        # Typically MOT files contain joint angles that need to be converted to SMPL pose parameters

        # For demonstration, we'll create placeholder parameters
        n_frames = len(self.marker_data)

        # Create default beta if not provided
        if beta is None:
            beta = torch.zeros(10, device=self.device)

        # Extract root translation and orientation from motion data
        # This is a placeholder - you'll need to adapt this to your specific MOT file format
        root_trans = torch.zeros((1, n_frames, 3), device=self.device)
        root_orient = torch.zeros((1, n_frames, 3), device=self.device)
        body_pose = torch.zeros((1, n_frames, 69), device=self.device)

        # Extract columns from motion data that correspond to root position and orientation
        # This assumes your MOT file has specific column names - adjust as needed
        if "pelvis_tx" in self.motion_data.columns:
            root_trans[0, :, 0] = torch.tensor(
                self.motion_data["pelvis_tx"].values, device=self.device
            )
            root_trans[0, :, 1] = torch.tensor(
                self.motion_data["pelvis_ty"].values, device=self.device
            )
            root_trans[0, :, 2] = torch.tensor(
                self.motion_data["pelvis_tz"].values, device=self.device
            )

        # Extract orientation (assuming Euler angles in the MOT file)
        if "pelvis_tilt" in self.motion_data.columns:
            # Convert Euler angles to axis-angle (simplified)
            # This is a placeholder - you'll need proper conversion based on your convention
            root_orient[0, :, 0] = torch.tensor(
                self.motion_data["pelvis_tilt"].values, device=self.device
            )
            root_orient[0, :, 1] = torch.tensor(
                self.motion_data["pelvis_list"].values, device=self.device
            )
            root_orient[0, :, 2] = torch.tensor(
                self.motion_data["pelvis_rotation"].values, device=self.device
            )

        # Extract body pose parameters from joint angles in MOT file
        # This is a complex mapping that depends on your specific MOT file format
        # You'll need to map OpenSim joint angles to SMPL pose parameters

        # Run SMPL forward pass
        smpl_output = pred_smpl(
            body_model=self.smpl_model,
            trans=root_trans,
            root_orient=root_orient,
            body_pose=body_pose,
            betas=beta.unsqueeze(0),
        )

        return {
            "vertices": smpl_output["points3d"],
            "joints": smpl_output["joints3d_op"],
            "trans": root_trans,
            "root_orient": root_orient,
            "body_pose": body_pose,
            "beta": beta,
        }

    def save_smpl_params(self, smpl_params, filename="smpl_params.pkl"):
        """
        Save SMPL parameters to a file

        Parameters:
        -----------
        smpl_params: dict
            Dictionary containing SMPL parameters
        filename: str
            Name of the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving SMPL parameters to {output_path}")

        # Convert tensors to numpy arrays for saving
        smpl_params_np = {}
        for key, value in smpl_params.items():
            if isinstance(value, torch.Tensor):
                smpl_params_np[key] = value.detach().cpu().numpy()
            else:
                smpl_params_np[key] = value

        with open(output_path, "wb") as f:
            pickle.dump(smpl_params_np, f)

        return output_path

    def generate_visualization(
        self, smpl_params, output_filename="smpl_visualization.mp4"
    ):
        """
        Generate a visualization of the SMPL model

        Parameters:
        -----------
        smpl_params: dict
            Dictionary containing SMPL parameters
        output_filename: str
            Name of the output video file

        Returns:
        --------
        output_path: str
            Path to the output video file
        """
        try:
            import pickle
            import os

            logger.info("Generating SMPL visualization")

            # Create a mock config object with the necessary attributes
            class MockConfig:
                def __init__(self, device):
                    self.DEVICE = device

            cfg = MockConfig(self.device)

            # Prepare results in the format expected by run_vis_on_demo
            results = {
                "0": {
                    "pose_world": np.concatenate(
                        [
                            smpl_params["root_orient"].cpu().numpy().squeeze(),
                            smpl_params["body_pose"].cpu().numpy().squeeze(),
                        ],
                        axis=1,
                    ),
                    "trans_world": smpl_params["trans"].cpu().numpy().squeeze(),
                    "betas": smpl_params["beta"].cpu().numpy(),
                    "verts": smpl_params["vertices"].cpu().numpy().squeeze(),
                    "frame_ids": np.arange(smpl_params["vertices"].shape[1]),
                }
            }

            # Create a WHAM-compatible SMPL model with faces
            from slahmr.slahmr.model.smpl import SMPL_Layer

            # Path to the SMPL model
            smpl_model_path = defaults()["SMPL_NEUTRAL_PATH"]

            # Load SMPL model that includes faces for visualization
            with open(smpl_model_path, "rb") as f:
                model_data = pickle.load(f, encoding="latin1")

            # Get faces from the model data
            faces = model_data["f"]

            # Create a simple SMPL wrapper with faces attribute for visualization
            class SMPLWithFaces:
                def __init__(self, faces):
                    self.faces = torch.tensor(faces).unsqueeze(0).to(self.device)

            smpl_vis = SMPLWithFaces(faces)

            # Create a blank video path (WHAM viz requires a video path)
            blank_video_path = os.path.join(self.output_dir, "blank_video.mp4")
            self._create_blank_video(blank_video_path, results)

            # Run visualization
            output_path = os.path.join(self.output_dir, output_filename)
            run_vis_on_demo(
                cfg,
                blank_video_path,
                results,
                self.output_dir,
                smpl_vis,
                vis_global=True,
            )

            # Rename the output file
            os.rename(os.path.join(self.output_dir, "output.mp4"), output_path)

            # Clean up temporary file
            if os.path.exists(blank_video_path):
                os.remove(blank_video_path)

            logger.info(f"SMPL visualization saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            logger.info("Falling back to simple visualization method")

            # Implement a simpler visualization method as fallback
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            output_path = os.path.join(
                self.output_dir, output_filename.replace(".mp4", ".png")
            )

            # Get vertices for a single frame (middle frame)
            n_frames = smpl_params["vertices"].shape[1]
            mid_frame = n_frames // 2
            verts = smpl_params["vertices"][0, mid_frame].cpu().numpy()

            # Create a simple 3D plot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Plot a subset of vertices for clarity
            step = 10  # Plot every 10th vertex
            ax.scatter(
                verts[::step, 0],
                verts[::step, 1],
                verts[::step, 2],
                c="b",
                marker="o",
                s=10,
            )

            # Set equal aspect ratio
            max_range = (
                np.array(
                    [
                        verts[:, 0].max() - verts[:, 0].min(),
                        verts[:, 1].max() - verts[:, 1].min(),
                        verts[:, 2].max() - verts[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )

            mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
            mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
            mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.title("SMPL Model Visualization (Frame {})".format(mid_frame))
            plt.savefig(output_path)

            logger.info(f"Simple SMPL visualization saved to {output_path}")
            return output_path

    def visualize_optimized_results(
        self, optimized_results_path, create_video=True, create_interactive=True
    ):
        """
        Visualize optimized results from a pickle file

        Parameters:
        -----------
        optimized_results_path: str
            Path to the optimized results file (.pkl)
        create_video: bool
            Whether to create a video animation (default: True)
        create_interactive: bool
            Whether to create an interactive HTML visualization (default: True)

        Returns:
        --------
        dict:
            Dictionary with paths to output files
        """
        try:
            # Generate SMPL vertices if not already present
            self._update_results_with_vertices(optimized_results_path)

            outputs = {}

            # Create video animation
            if create_video:
                video_path = self.create_animation_video(optimized_results_path)
                outputs["video"] = video_path

            # Create interactive visualization
            if create_interactive:
                # Create the interactive single view (lightweight)
                interactive_path = self.create_interactive_single_view(
                    optimized_results_path
                )
                outputs["interactive"] = interactive_path

                # Optionally, create the multi-view visualization (heavier)
                # multi_view_path = self.create_interactive_visualization(optimized_results_path)
                # outputs['multi_view'] = multi_view_path

            return outputs

        except Exception as e:
            logger.error(f"Error visualizing optimized results: {str(e)}")
            return {}

    def _generate_simple_visualization_from_pkl(self, pkl_path, output_filename):
        """Generate a simple visualization from a PKL file"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import pickle

            output_path = os.path.join(self.output_dir, output_filename)

            # Load the data
            with open(pkl_path, "rb") as f:
                results = pickle.load(f)

            # Get the first subject
            subject_id = list(results.keys())[0]
            verts = results[subject_id]["verts"]

            # If verts is None, we can't visualize
            if verts is None:
                logger.error(
                    "No vertices found in the optimized results. Cannot visualize."
                )
                return None

            # Get vertices for a single frame (middle frame)
            n_frames = len(results[subject_id]["frame_ids"])
            mid_frame = n_frames // 2

            # Create a simple 3D plot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Plot a subset of vertices for clarity
            step = 10  # Plot every 10th vertex
            ax.scatter(
                verts[mid_frame, ::step, 0],
                verts[mid_frame, ::step, 1],
                verts[mid_frame, ::step, 2],
                c="b",
                marker="o",
                s=10,
            )

            # Set equal aspect ratio
            max_range = (
                np.array(
                    [
                        verts[mid_frame, :, 0].max() - verts[mid_frame, :, 0].min(),
                        verts[mid_frame, :, 1].max() - verts[mid_frame, :, 1].min(),
                        verts[mid_frame, :, 2].max() - verts[mid_frame, :, 2].min(),
                    ]
                ).max()
                / 2.0
            )

            mid_x = (verts[mid_frame, :, 0].max() + verts[mid_frame, :, 0].min()) * 0.5
            mid_y = (verts[mid_frame, :, 1].max() + verts[mid_frame, :, 1].min()) * 0.5
            mid_z = (verts[mid_frame, :, 2].max() + verts[mid_frame, :, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.title("Optimized SMPL Model (Frame {})".format(mid_frame))
            plt.savefig(output_path)

            logger.info(f"Simple optimized visualization saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating simple visualization: {str(e)}")
            return None

    def _generate_simple_visualization(
        self, smpl_params, output_filename="simple_smpl_visualization.png"
    ):
        """Generate a simple visualization using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            output_path = os.path.join(self.output_dir, output_filename)

            # Get vertices for a single frame (middle frame)
            n_frames = smpl_params["vertices"].shape[1]
            mid_frame = n_frames // 2
            verts = smpl_params["vertices"][0, mid_frame].cpu().numpy()

            # Create a simple 3D plot
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Plot a subset of vertices for clarity
            step = 10  # Plot every 10th vertex
            ax.scatter(
                verts[::step, 0],
                verts[::step, 1],
                verts[::step, 2],
                c="b",
                marker="o",
                s=10,
            )

            # Set equal aspect ratio
            max_range = (
                np.array(
                    [
                        verts[:, 0].max() - verts[:, 0].min(),
                        verts[:, 1].max() - verts[:, 1].min(),
                        verts[:, 2].max() - verts[:, 2].min(),
                    ]
                ).max()
                / 2.0
            )

            mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
            mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
            mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.title("SMPL Model Visualization (Frame {})".format(mid_frame))
            plt.savefig(output_path)

            logger.info(f"Simple SMPL visualization saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating simple visualization: {str(e)}")
            return None

    def _create_blank_video(self, video_path, results):
        """Create a blank video file for visualization purposes"""
        import cv2
        import numpy as np

        # Determine number of frames from the results
        id_0 = list(results.keys())[0]
        n_frames = len(results[id_0]["frame_ids"])

        # Create a blank video
        width, height = 1920, 1080
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write blank frames
        blank_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        for _ in range(n_frames):
            video_writer.write(blank_frame)

        video_writer.release()

    def create_animation_video(
        self, optimized_results_path, output_filename="smpl_animation.mp4"
    ):
        """
        Create a video animation of the SMPL model with full body visibility
        """
        try:
            import os
            import pickle
            import numpy as np
            import torch
            import cv2
            from tqdm import tqdm

            # Import WHAM's renderer directly
            from WHAM.lib.vis.renderer import Renderer

            logger.info(
                f"Creating animation video using WHAM's renderer: {optimized_results_path}"
            )

            # Load the optimized results
            with open(optimized_results_path, "rb") as f:
                results = pickle.load(f)

            # Get vertices
            subject_id = list(results.keys())[0]
            verts = results[subject_id]["verts"]

            # If vertices don't exist, we need to run visualization first to generate them
            if verts is None:
                logger.info(
                    "No vertices found in results. Running visualization first to generate vertices."
                )
                self._update_results_with_vertices(optimized_results_path)

                # Reload the results with vertices
                with open(optimized_results_path, "rb") as f:
                    results = pickle.load(f)
                subject_id = list(results.keys())[0]
                verts = results[subject_id]["verts"]

            # Get faces for the SMPL model
            smpl_model_path = defaults()["SMPL_NEUTRAL_PATH"]
            with open(smpl_model_path, "rb") as f:
                model_data = pickle.load(f, encoding="latin1")
            faces = model_data["f"]

            # Set up paths
            output_path = os.path.join(self.output_dir, output_filename)

            # Set up device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Set up video dimensions
            width, height = 800, 800

            # Create a renderer with a much smaller focal length for a wider view
            focal_length = 500  # Reduced from 1000 to 500 for a wider field of view
            renderer = Renderer(width, height, focal_length, device, faces)

            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Create a white background
            background = np.ones((height, width, 3), dtype=np.uint8) * 255  # Pure white

            # Process frames
            n_frames = verts.shape[0]
            skip_frames = max(1, n_frames // 100)  # Limit to ~100 frames for speed

            logger.info(
                f"Rendering {n_frames//skip_frames} frames from {n_frames} total"
            )

            # Pre-compute global bounds for consistent scale
            all_verts = torch.tensor(
                verts.reshape(-1, 3), dtype=torch.float32, device=device
            )

            # Set a fixed camera view from the front with more distance
            R = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
            ).unsqueeze(0)

            # Move camera farther back for a wider view
            T = torch.tensor([0.0, 0.0, -5.0], device=device).unsqueeze(0)

            # Set up camera once
            renderer.create_camera(R, T)

            # Set global bounding box with a very small scale to ensure full visibility
            renderer.update_bbox(
                all_verts, scale=0.5
            )  # Much smaller scale to show full body

            # Store the bounding box
            bbox = renderer.bboxes.clone()

            for frame_idx in tqdm(
                range(0, n_frames, skip_frames), desc="Rendering frames"
            ):
                # Create a clean background for each frame
                current_bg = background.copy()

                # Get the vertices for this frame and convert to tensor
                frame_verts = torch.tensor(
                    verts[frame_idx], dtype=torch.float32, device=device
                )

                # Force our fixed bbox
                renderer.bboxes = bbox.clone()

                # Use a distinctive red color for better visibility
                model_color = [0.9, 0.2, 0.2]  # Red color often stands out well

                # Render mesh onto background
                img = renderer.render_mesh(frame_verts, current_bg, colors=model_color)

                # Add frame information
                cv2.putText(
                    img,
                    f"Frame: {frame_idx}/{n_frames}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                )

                # Write to video
                video_writer.write(img)

            # Clean up
            video_writer.release()

            logger.info(f"Animation video saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating animation video: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

            # Fall back to the simple visualization with error handling
            return self._generate_simple_visualization_from_pkl(
                optimized_results_path, output_filename.replace(".mp4", ".png")
            )

    def create_interactive_visualization(
        self, optimized_results_path, output_filename="smpl_interactive.html"
    ):
        """
        Create an interactive 3D visualization of the SMPL model using Plotly
        """
        try:
            import os
            import pickle
            import numpy as np
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            logger.info(
                f"Creating interactive 3D visualization: {optimized_results_path}"
            )

            # Load the optimized results
            with open(optimized_results_path, "rb") as f:
                results = pickle.load(f)

            # Get vertices
            subject_id = list(results.keys())[0]
            verts = results[subject_id]["verts"]

            # If vertices don't exist, we need to run visualization first to generate them
            if verts is None:
                logger.info(
                    "No vertices found in results. Running visualization first to generate vertices."
                )
                self.visualize_optimized_results(optimized_results_path)

                # Reload the results with vertices
                with open(optimized_results_path, "rb") as f:
                    results = pickle.load(f)
                subject_id = list(results.keys())[0]
                verts = results[subject_id]["verts"]

            # Get faces for the SMPL model
            smpl_model_path = defaults()["SMPL_NEUTRAL_PATH"]
            with open(smpl_model_path, "rb") as f:
                model_data = pickle.load(f, encoding="latin1")
            faces = model_data["f"]

            # Set up paths
            output_path = os.path.join(self.output_dir, output_filename)

            # Select a subset of frames to display (for performance reasons)
            n_frames = verts.shape[0]
            frame_step = max(1, n_frames // 10)  # Show up to 10 frames
            selected_frames = list(range(0, n_frames, frame_step))

            # Create subplots - one for each selected frame
            n_cols = min(3, len(selected_frames))
            n_rows = (len(selected_frames) + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                specs=[
                    [{"type": "mesh3d"} for _ in range(n_cols)] for _ in range(n_rows)
                ],
                subplot_titles=[f"Frame {idx}" for idx in selected_frames],
            )

            # Calculate overall scale for consistent visualization
            all_verts = verts.reshape(-1, 3)
            mean_vert = np.mean(all_verts, axis=0)
            max_range = np.max(np.ptp(all_verts, axis=0)) / 2

            # Add each frame as a mesh
            row, col = 1, 1
            for i, frame_idx in enumerate(selected_frames):
                # Get vertices for this frame
                frame_verts = verts[frame_idx]

                # Create a mesh plot
                mesh = go.Mesh3d(
                    x=frame_verts[:, 0],
                    y=frame_verts[:, 1],
                    z=frame_verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color="lightblue",
                    opacity=0.8,
                    name=f"Frame {frame_idx}",
                )

                fig.add_trace(mesh, row=row, col=col)

                # Update layout for this subplot
                camera = dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=mean_vert[0], y=mean_vert[1], z=mean_vert[2]),
                )

                fig.update_layout(**{f"scene{i+1}_camera": camera})
                fig.update_layout(**{f"scene{i+1}_aspectmode": "data"})

                # Update row and column for next subplot
                col += 1
                if col > n_cols:
                    col = 1
                    row += 1

            # Set up overall layout
            fig.update_layout(
                title="SMPL Model Visualization",
                width=1200,
                height=300 * n_rows,
                margin=dict(l=0, r=0, b=0, t=40),
            )

            # Create animation frame for interactive slider (optional)
            frames = []
            for frame_idx in range(0, n_frames, frame_step):
                frame_data = []
                for subplot_idx in range(len(selected_frames)):
                    # Create a frame for each subplot
                    if frame_idx < n_frames:
                        frame_verts = verts[frame_idx]
                        frame_data.append(
                            go.Mesh3d(
                                x=frame_verts[:, 0],
                                y=frame_verts[:, 1],
                                z=frame_verts[:, 2],
                                i=faces[:, 0],
                                j=faces[:, 1],
                                k=faces[:, 2],
                                color="lightblue",
                                opacity=0.8,
                            )
                        )
                    else:
                        # If we've run out of frames, repeat the last one
                        frame_data.append(
                            go.Mesh3d(
                                x=verts[-1][:, 0],
                                y=verts[-1][:, 1],
                                z=verts[-1][:, 2],
                                i=faces[:, 0],
                                j=faces[:, 1],
                                k=faces[:, 2],
                                color="lightblue",
                                opacity=0.8,
                            )
                        )
                frames.append(go.Frame(data=frame_data, name=f"frame_{frame_idx}"))

            # Save the figure to HTML
            fig.write_html(output_path, auto_open=False)

            logger.info(f"Interactive visualization saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating interactive visualization: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def create_interactive_single_view(
        self, optimized_results_path, output_filename="smpl_viewer.html"
    ):
        """
        Create a lightweight interactive 3D visualization of the SMPL model with a slider to animate frames
        """
        try:
            import os
            import pickle
            import numpy as np
            import plotly.graph_objects as go

            logger.info(
                f"Creating lightweight interactive 3D visualization: {optimized_results_path}"
            )

            # Load the optimized results
            with open(optimized_results_path, "rb") as f:
                results = pickle.load(f)

            # Get vertices
            subject_id = list(results.keys())[0]
            verts = results[subject_id]["verts"]

            # If vertices don't exist, we need to run visualization first to generate them
            if verts is None:
                logger.info(
                    "No vertices found in results. Running visualization first to generate vertices."
                )
                self._update_results_with_vertices(optimized_results_path)

                # Reload the results with vertices
                with open(optimized_results_path, "rb") as f:
                    results = pickle.load(f)
                subject_id = list(results.keys())[0]
                verts = results[subject_id]["verts"]

            # Get faces for the SMPL model
            smpl_model_path = defaults()["SMPL_NEUTRAL_PATH"]
            with open(smpl_model_path, "rb") as f:
                model_data = pickle.load(f, encoding="latin1")
            faces = model_data["f"]

            # Set up paths
            output_path = os.path.join(self.output_dir, output_filename)

            # Select a subset of frames to display (for performance reasons)
            n_frames = verts.shape[0]
            frame_step = max(1, n_frames // 50)  # Show up to 50 frames
            selected_frames = list(range(0, n_frames, frame_step))

            # Calculate global bounds for consistent scaling across all frames
            all_verts = verts.reshape(-1, 3)
            mean_vert = np.mean(all_verts, axis=0)

            # Calculate the range for each dimension
            x_range = [np.min(all_verts[:, 0]), np.max(all_verts[:, 0])]
            y_range = [np.min(all_verts[:, 1]), np.max(all_verts[:, 1])]
            z_range = [np.min(all_verts[:, 2]), np.max(all_verts[:, 2])]

            # Add padding to ranges
            padding = 0.2  # 20% padding
            x_pad = (x_range[1] - x_range[0]) * padding
            y_pad = (y_range[1] - y_range[0]) * padding
            z_pad = (z_range[1] - z_range[0]) * padding

            x_range = [x_range[0] - x_pad, x_range[1] + x_pad]
            y_range = [y_range[0] - y_pad, y_range[1] + y_pad]
            z_range = [z_range[0] - z_pad, z_range[1] + z_pad]

            # Create initial mesh for the first frame
            frame_verts = verts[0]

            # Create a figure with a single mesh plot
            fig = go.Figure(
                data=[
                    go.Mesh3d(
                        x=frame_verts[:, 0],
                        y=frame_verts[:, 1],
                        z=frame_verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color="royalblue",
                        opacity=0.8,
                        name="SMPL Model",
                        lighting=dict(
                            ambient=0.8,
                            diffuse=0.9,
                            specular=0.5,
                            roughness=0.5,
                            fresnel=0.8,
                        ),
                    )
                ],
                layout=go.Layout(
                    title=f"SMPL Model Visualization",
                    updatemenus=[
                        {
                            "type": "buttons",
                            "buttons": [
                                {
                                    "label": "Play",
                                    "method": "animate",
                                    "args": [
                                        None,
                                        {
                                            "frame": {"duration": 100, "redraw": True},
                                            "fromcurrent": True,
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                    sliders=[
                        {
                            "currentvalue": {"prefix": "Frame: "},
                            "steps": [
                                {
                                    "method": "animate",
                                    "label": str(i),
                                    "args": [
                                        [f"frame_{i}"],
                                        {
                                            "frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                }
                                for i in selected_frames
                            ],
                        }
                    ],
                ),
            )

            # Set up consistent camera view and fixed axes
            fig.update_layout(
                scene=dict(
                    # Fixed camera view
                    camera=dict(eye=dict(x=2.0, y=0.5, z=0.5), up=dict(x=0, y=1, z=0)),
                    # Fixed axis ranges
                    xaxis=dict(
                        range=x_range,
                        autorange=False,
                        showticklabels=False,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        showbackground=False,
                        title="",
                    ),
                    yaxis=dict(
                        range=y_range,
                        autorange=False,
                        showticklabels=False,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        showbackground=False,
                        title="",
                    ),
                    zaxis=dict(
                        range=z_range,
                        autorange=False,
                        showticklabels=False,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        showbackground=False,
                        title="",
                    ),
                    # Ensures the aspect ratio is consistent
                    aspectmode="cube",
                ),
                # Overall figure sizing
                width=800,
                height=800,
                margin=dict(l=0, r=0, b=0, t=30),
                paper_bgcolor="rgba(240, 240, 240, 1)",
                template="plotly_white",
            )

            # Create animation frames
            frames = []
            for i, frame_idx in enumerate(selected_frames):
                frame_verts = verts[frame_idx]
                frames.append(
                    go.Frame(
                        data=[
                            go.Mesh3d(
                                x=frame_verts[:, 0],
                                y=frame_verts[:, 1],
                                z=frame_verts[:, 2],
                                i=faces[:, 0],
                                j=faces[:, 1],
                                k=faces[:, 2],
                                color="royalblue",
                                opacity=0.8,
                                lighting=dict(
                                    ambient=0.8,
                                    diffuse=0.9,
                                    specular=0.5,
                                    roughness=0.5,
                                    fresnel=0.8,
                                ),
                            )
                        ],
                        name=f"frame_{frame_idx}",
                        # No layout changes to maintain consistent view
                    )
                )

            fig.frames = frames

            # Save the figure to HTML with full HTML wrapper (self-contained)
            fig.write_html(output_path, auto_open=False, include_plotlyjs="cdn")

            logger.info(f"Interactive visualization saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating interactive visualization: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def predict_smpl_from_markers(self):
        """
        Predict SMPL models from marker data and motion file

        Returns:
        --------
        dict:
            Dictionary containing paths to output files
        """
        try:
            # Preprocess marker data to match SMPL model
            markers_smpl, _ = self._preprocess_markers()

            # Predict SMPL parameters
            optimized_pkl_path = self._predict_smpl_params(markers_smpl)

            # Save results
            logger.info("SMPL prediction complete.")
            logger.info(f"Results saved to {optimized_pkl_path}")

            # Visualize results
            visualization_results = self.visualize_optimized_results(
                optimized_pkl_path, create_video=False, create_interactive=False
            )

            # Print visualization outputs
            if "video" in visualization_results:
                logger.info(
                    f"Video animation saved to: {visualization_results['video']}"
                )
            if "interactive" in visualization_results:
                logger.info(
                    f"Interactive 3D visualization saved to: {visualization_results['interactive']}"
                )
                logger.info(
                    "Open the HTML file in a web browser to interact with the 3D model"
                )

            # Return paths to output files
            output_files = {
                "optimized_results": optimized_pkl_path,
                **visualization_results,
            }

            return output_files

        except Exception as e:
            logger.error(f"Error predicting SMPL from markers: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return {}

    def _update_results_with_vertices(self, optimized_results_path):
        """
        Update optimized results with SMPL vertices if not already present

        Parameters:
        -----------
        optimized_results_path: str
            Path to the optimized results file (.pkl)
        """
        import pickle

        # Load optimized results
        with open(optimized_results_path, "rb") as f:
            results = pickle.load(f)

        # Check if results contain vertices
        subject_id = list(results.keys())[0]
        if "verts" not in results[subject_id] or results[subject_id]["verts"] is None:
            logger.info("Computing SMPL vertices from parameters...")

            # Get pose and shape parameters
            pose = results[subject_id]["pose_world"]
            betas = results[subject_id]["betas"]
            trans = results[subject_id]["trans_world"]

            # Convert to torch tensors
            pose_tensor = torch.tensor(pose, device=self.device)
            betas_tensor = torch.tensor(betas, device=self.device)
            trans_tensor = torch.tensor(trans, device=self.device)

            # Forward pass through SMPL model
            output = self.smpl_model(
                betas=betas_tensor,
                body_pose=pose_tensor[:, 3:],
                global_orient=pose_tensor[:, :3],
                transl=trans_tensor,
            )

            # Get vertices
            verts = output.vertices.cpu().numpy()

            # Update results
            results[subject_id]["verts"] = verts

            # Save updated results
            with open(optimized_results_path, "wb") as f:
                pickle.dump(results, f)

            logger.info(f"Updated results with vertices: {optimized_results_path}")
