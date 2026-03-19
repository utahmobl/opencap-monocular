import os
import traceback
from loguru import logger
import utils.utils_optim as ut
from utils.optimization_formulations import OptimizeExtrinsics, OptimizePose
from utils.utils_activity_classification import predict_activity_from_video
from utils.utils_vis import (
    plot_objective_function,
    plot_3d_keypoints_interactive_plotly,
    plot_2d_keypoints_interactive_plotly,
)
from utils.opensim import utils_opensim
from utils.utilsCameraPy3 import rotateIntrinsics, getVideoRotation
import numpy as np
import torch
import webbrowser
import pickle
import joblib
from utils.utilsChecker import detectGait
import yaml


def run_optimization(
    data_dir: str,  # Directory containing input data
    trial_name: str,  # Name of the trial
    height_m: float,  # Subject height in meters
    mass_kg: float,  # Subject mass in kg
    sex: str,  # Subject sex ('m' or 'f')
    intrinsics_pth: str,  # Path to camera intrinsics file
    case: str = "5",  # Case identifier
    trc_rot: dict = {"x": 0, "y": 65, "z": 0},  # Rotation for TRC export
    run_opensim_original_wham: bool = True,  # Run OpenSim on original WHAM results
    run_opensim_opt2: bool = True,  # Run OpenSim on optimized results
    use_gpu: bool = True,  # Use GPU for optimization
    filter_freq: int = 6,  # Cutoff frequency for filtering
    static_cam: bool = True,  # Use static camera assumption
    optimize_camera: bool = True,  # Optimize camera parameters in second stage
    n_iter_opt2: int = 75,  # Number of iterations for optimization 2
    print_loss_terms: bool = False,  # Print individual loss terms
    plotting: bool = False,  # Generate plots
    smoothness_diff_n: int = 1,  # Order of differentiation for smoothness
    save_video_debug: bool = True,  # Save debug video
    output_path: str = None,  # Path for output files
    video_path: str = None,  # Path to video file
    activity: str = None,  # Activity to optimize for
    save_smpl_for_viz: bool = False,  # Save SMPL for visualization
    weights_opt2: dict = None,  # Custom weights for optimization (for HP search)
    rotation: int = None,  # Video rotation in degrees (None means no rotation)
    create_contact_visualizations: bool = False,  # Whether to create contact probability plots and video overlay
):
    """
    Run the pose optimization pipeline.

    This function takes WHAM reconstruction results and optimizes both camera parameters
    and human pose to improve reconstruction accuracy. The results can be exported
    to OpenSim for biomechanical analysis.

    Returns:
        dict: A dictionary containing paths to various output files, e.g.,
              {'output_dir': str, 'trc_file': str, 'scaled_model_file': str,
               'ik_results_file': str, 'optimized_pkl': str,
               'debug_video': str | None, 'trimmed_video': str | None}
    """

    output_paths = {  # Initialize dictionary to store paths
        "output_dir": output_path,
        "trc_file": None,
        "scaled_model_file": None,
        "ik_results_file": None,
        "optimized_pkl": None,
        "debug_video": None,
        "trimmed_video": None,
        "plot_objective": None,
        "plot_2d": None,
        "plot_3d": None,  # Assuming 3D plot might be added later or wasn't returning path
        "wham_output_pkl": None,
        "keypoints_3d_cam_pkl": None,
        "vertices_3d_cam_pkl": None,
    }

    torch.cuda.empty_cache()

    global r_root_in_world, t_root_in_world, key2d_smpl_opt1, body_pose

    # Find device
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        print("Using GPU for optimization.")
    else:
        device = "cpu"
        if use_gpu:
            print("No GPU available, using CPU.")

    # Load WHAM results
    wham_regression_results_pth = os.path.join(data_dir, "wham_output.pkl")
    try:
        wham_regression_results = ut.load_results(wham_regression_results_pth)
        output_paths["wham_output_pkl"] = wham_regression_results_pth
    except Exception as e:
        # go one level down, repeat the last element of the data_dir
        data_dir = data_dir.rsplit("/", 1)[0]
        wham_regression_results_pth = os.path.join(data_dir, "wham_output.pkl")
        wham_regression_results = ut.load_results(wham_regression_results_pth)
        output_paths["wham_output_pkl"] = wham_regression_results_pth
    except Exception as e:
        logger.error(f"Error loading WHAM results: {e}")
        breakpoint()

    # Load intrinsics from file
    intrinsics = ut.load_intrinsics(intrinsics_pth)

    cap, frame_rate, n_frames = None, None, None

    # Construct path to original video
    if video_path is None:
        base_path = data_dir.split("output/")[0]
        subject_num = data_dir.split("subject")[1].split("/")[0]
        subject = f"subject{subject_num}"
        session_num = data_dir.split("Session")[1].split("/")[0]
        session = f"Session{session_num}"
        camera_num = data_dir.split("Cam")[1].split("/")[0]
        camera = f"Cam{camera_num}"
        movement = data_dir.split("/")[-1]

        original_video_folder = os.path.join(
            base_path,
            "LabValidation_withVideos1",
            subject,
            "VideoData",
            session,
            camera,
            movement,
        )
        cap, frame_rate, n_frames, video_path = ut.get_video_info(
            data_dir=original_video_folder, release=False
        )
    else:
        original_video_folder = video_path
        cap, frame_rate, n_frames, _ = ut.get_video_info(
            video_path=video_path, release=False
        )

    frame_rate = round(frame_rate, 0)
    logger.info(f"Frame rate: {frame_rate}")

    if rotation is None:
        rotation = getVideoRotation(video_path)
    else:
        logger.info(f"Rotation provided: {rotation}")

    # --- Resolution scaling -----------------------------------------------
    # All intrinsics pickles are calibrated at 720p portrait (720×1280).
    # If the video was recorded at a different resolution we need to scale
    # fx, fy, cx, cy proportionally before doing anything else.
    import cv2

    if cap is not None:
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if "calib_portrait_h" in intrinsics and "calib_portrait_w" in intrinsics:
            calib_h = intrinsics.pop("calib_portrait_h")
            calib_w = intrinsics.pop("calib_portrait_w")
            # Express the actual video dimensions in portrait orientation so
            # we can compare apples-to-apples with the portrait calibration.
            if rotation in [0, 180]:
                # Landscape video: portrait_h = W_landscape, portrait_w = H_landscape
                actual_portrait_h = float(vid_width)
                actual_portrait_w = float(vid_height)
            else:
                # Portrait / upside-down video: dims already in portrait space
                actual_portrait_h = float(vid_height)
                actual_portrait_w = float(vid_width)

            scale_x = actual_portrait_w / calib_w
            scale_y = actual_portrait_h / calib_h
            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                logger.info(
                    f"Scaling intrinsics from calibration portrait resolution "
                    f"{calib_w:.0f}x{calib_h:.0f} → actual portrait "
                    f"{actual_portrait_w:.0f}x{actual_portrait_h:.0f} "
                    f"(scale x={scale_x:.3f}, y={scale_y:.3f})"
                )
                intrinsics["fx"] *= scale_x
                intrinsics["fy"] *= scale_y
                intrinsics["cx"] *= scale_x
                intrinsics["cy"] *= scale_y
    # ----------------------------------------------------------------------

    if rotation in [0, 180]:
        horizontal = True
        if cap is not None:
            # rotateIntrinsics expects portrait dimensions [H_portrait, W_portrait].
            # For a landscape video (rotation 0/180), the portrait frame is the
            # 90°-rotated version: H_portrait = W_landscape, W_portrait = H_landscape.
            image_size = [
                vid_width,
                vid_height,
            ]  # portrait dims: [H_portrait, W_portrait]
        else:
            image_size = None
        intrinsics = rotateIntrinsics(intrinsics, rotation, imageSize=image_size)
        logger.info("Intrinsics rotated")

    # Load WHAM results - handle multiple people if detected
    num_people = len(wham_regression_results)
    if num_people > 1:
        logger.warning(f"More than one person detected: {num_people}")
        for i in range(num_people):
            wham_result_i = wham_regression_results[i]
            if len(wham_result_i) > 0:
                wham_result = wham_result_i
                break
    else:
        wham_result = wham_regression_results[0]

    # Filter contact information
    try:
        wham_result["contact"] = ut.filter_array(
            wham_result["contact"], order=4, cutoff_freq=6, sampling_rate=frame_rate
        )
    except Exception as e:
        logger.error(f"Error filtering contact: {e}. At {wham_regression_results_pth}")

    # Convert numpy arrays to tensors
    wham_result = {
        k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
        for k, v in wham_result.items()
    }

    # Convert keypoints from float64 to float32
    try:
        wham_result["tracking_results_for_reproj"]["keypoints"] = wham_result[
            "tracking_results_for_reproj"
        ]["keypoints"].astype(np.float32)
    except Exception as e:
        print(e)
        breakpoint()

    # Get 2D keypoints from OpenPose
    key2d = ut.get_openpose_keypoints(
        wham_result["tracking_results_for_reproj"]["keypoints"],
        filter_freq=filter_freq,
        sample_rate=frame_rate,
        device=device,
    )

    # TODO check if this is correct and adapt for mono range
    # Create frame ranges from WHAM results
    # Get frame IDs from wham_result (could be "frame_id" or "frame_ids")
    if "frame_id" in wham_result:
        frame_ids = wham_result["frame_id"]
    elif "frame_ids" in wham_result:
        frame_ids = wham_result["frame_ids"]
    else:
        logger.warning("No frame_id found in wham_result, using sequential indices")
        n_wham_frames = len(wham_result["trans_world"])
        frame_ids = np.arange(n_wham_frames)

    # opencap_mono_frame_range_wham_ref: indices into wham_result arrays (0 to n_frames-1)
    n_wham_frames = len(frame_ids)
    opencap_mono_frame_range_wham_ref = range(n_wham_frames)

    # opencap_mono_frame_range_video_ref: actual video frame numbers from first to last frame_id
    first_frame_id = int(frame_ids[0])
    last_frame_id = int(frame_ids[-1])
    opencap_mono_frame_range_video_ref = range(first_frame_id, last_frame_id + 1)

    logger.info(
        f"WHAM frame range (wham reference): {opencap_mono_frame_range_wham_ref}"
    )
    logger.info(
        f"WHAM frame range (video reference): {opencap_mono_frame_range_video_ref}"
    )

    # Save trimmed video if requested
    save_trimmed_video = True
    if save_trimmed_video:
        # Save a trimmed video
        trimmed_video_path = ut.save_trimmed_video(
            data_dir,
            opencap_mono_frame_range_video_ref,
            original_video_folder,
            ffmpeg=True,
        )
        output_paths["trimmed_video"] = trimmed_video_path

    # Compute mean body shape parameters
    beta = ut.compute_mean_beta(wham_result["betas"])

    # Create SMPL model
    gender = "female" if sex == "f" else "male"
    smpl_model, _ = ut.load_smpl(device=device, gender=gender)

    # Preallocate tensors for camera extrinsics and 3D keypoints
    n_frames = len(opencap_mono_frame_range_wham_ref)
    r_world_to_cam = torch.empty(wham_result["poses_root_world"].shape, device=device)
    t_world_to_cam = torch.empty(wham_result["trans_world"].shape, device=device)
    key3d_op_smpl = torch.empty(n_frames, 25, 3, device=device)
    r_world_to_cam = r_world_to_cam[opencap_mono_frame_range_wham_ref, :, :]
    t_world_to_cam = t_world_to_cam[opencap_mono_frame_range_wham_ref, :]

    # The camera translations are bad at the edges because the model enters/exits and is very small
    for i_frame in range(n_frames):
        # Construct initial SMPL parameter tensors
        trans_world = wham_result["trans_world"][i_frame, :]
        root_orient_world = wham_result["poses_root_world"][i_frame, :, :]
        poses_body = wham_result["poses_body"][i_frame, :, :, :]

        # Handle static camera assumption
        if static_cam:  # use first frame
            cam_i_frame = 0  # Using first frame for static camera
        else:
            cam_i_frame = i_frame

        if i_frame == cam_i_frame:
            # Get camera extrinsics by assuming the root is the same in both camera and world frames
            # This is an initial guess that will be refined during optimization
            r_world_to_root = wham_result["poses_root_world"][cam_i_frame, :, :]
            r_cam_to_root = wham_result["poses_root_cam"][cam_i_frame, :, :]
            r_world_to_cam[cam_i_frame, :, :] = torch.matmul(
                r_world_to_root, r_cam_to_root.T
            )
            t_world_to_cam[cam_i_frame, :] = wham_result["trans_world"][
                cam_i_frame, :
            ] - torch.matmul(
                r_world_to_cam[cam_i_frame, :, :],
                wham_result["trans_cam"][cam_i_frame, :],
            )

        # Run SMPL forward to get 3D keypoints
        smpl_result = ut.pred_smpl(
            smpl_model,
            trans=wham_result["trans_world"][i_frame, :].reshape((1, 1, 3)),
            root_orient=wham_result["pose_world"][i_frame, :3].reshape((1, 1, 3)),
            body_pose=wham_result["pose_world"][i_frame, 3:].reshape((1, 1, -1)),
            betas=beta.reshape((1, len(beta))),
        )

        key3d_op_smpl[i_frame, :, :] = smpl_result["joints3d_op"]

    # When static_cam is True, broadcast first frame's camera to all frames
    # (only frame 0 was filled above; other indices were uninitialized)
    if static_cam and n_frames > 1:
        r_world_to_cam = r_world_to_cam[0:1, :, :].expand(n_frames, -1, -1).clone()
        t_world_to_cam = t_world_to_cam[0:1, :].expand(n_frames, -1).clone()

    # Run OpenSim on original WHAM results if requested
    if run_opensim_original_wham:
        # Initialize SMPL to Opensim converter for initial WHAM results
        smpl2osim_wham = utils_opensim.SMPL_to_Opensim(
            smpl_model,
            root_trans=wham_result["trans_world"],
            root_orient=wham_result["pose_world"][:, :3],
            body_pose=wham_result["pose_world"][:, 3:],
            beta=beta,
            output_dir=data_dir,
            trial_name=trial_name + "_" + case + "_wham_result",
            frame_rate=frame_rate,
            mass=mass_kg,
        )

        # Write TRC file
        logger.info("Writing TRC initial wham result.")
        smpl2osim_wham.smpl_to_trc(rotations=trc_rot)

        # Scale Opensim Model
        logger.info("Scaling model initial wham result.")
        smpl2osim_wham.scale_model()

        # Run IK
        logger.info("Running IK initial wham result.")
        smpl2osim_wham.run_ik()

    # Detect gait from the motion
    # Get ankle joint indices (right and left)
    ankle_indices = [11, 14]  # OpenPose ankle joint indices

    # Get vertical velocities of ankles
    ankle_positions = key3d_op_smpl[:, ankle_indices, :].cpu().numpy()
    ankle_velocities = np.diff(ankle_positions, axis=0)

    # Detect if it's gait
    is_gait = detectGait(
        ankle_velocities[:, 0, 1],  # Right ankle vertical velocity
        ankle_velocities[:, 1, 1],  # Left ankle vertical velocity
        frame_rate,
    )

    logger.info(f"Gait detected using vertical ankle velocities: {is_gait}")

    # Predict activity from video if not provided
    predicted_activity = None
    activity_detection_method = None
    flat_floor = False

    if activity is not None:
        predicted_activity = activity
        activity_detection_method = "user_provided"
        flat_floor = (
            True
            if activity.lower() == "squat"
            or activity.lower() == "sts"
            or activity.lower() == "walking"
            else False
        )
    else:
        try:
            predicted_activity, flat_floor = predict_activity_from_video(video_path)
            logger.info(
                f"Predicted activity using Video Activity Classifier: {predicted_activity}"
            )
            if predicted_activity is not None:
                activity_detection_method = "video_classifier"
                if "walking" in predicted_activity.lower():
                    is_gait = True
        except Exception as e:
            logger.error(f"Error predicting activity: {e}")
            predicted_activity = None
            flat_floor = False

    # Load parameters from YAML file
    with open("params/parameters.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Use provided weights for HP search if available, otherwise load from params
    if weights_opt2 is not None:
        logger.info("Using provided weights for hyperparameter search.")
    else:
        predicted_activity_lower = (
            predicted_activity.lower() if predicted_activity is not None else ""
        )

        if is_gait:
            weights_opt2 = params["weights_opt2_walking"]
            filter_freq = params["filter_freq"]["walking"]
            logger.info("Using walking parameters.")
            if predicted_activity is None:
                # Confirmed by ankle-velocity heuristic, not video classifier
                predicted_activity = "walking"
                activity_detection_method = "ankle_velocity_heuristic"
        elif "squat" in predicted_activity_lower:
            weights_opt2 = params["weights_opt2_squats"]
            filter_freq = params["filter_freq"]["squats"]
            logger.info("Using squat parameters.")
        elif (
            "sit-to-stand" in predicted_activity_lower
            or "sts" in predicted_activity_lower
        ):
            weights_opt2 = params["weights_opt2_sts"]
            filter_freq = params["filter_freq"]["STS"]
            logger.info("Using STS parameters.")
        else:
            weights_opt2 = params["weights_opt2_other"]
            filter_freq = params["filter_freq"]["other"]
            logger.info("Using other parameters, no flat floor loss.")
            if predicted_activity is None:
                predicted_activity = "other"
                activity_detection_method = "fallback_other"

    # Optimization 1: Camera extrinsics
    logger.info("Optimizing extrinsics.")
    try:
        # Initialize and run camera extrinsics optimization
        optimizer = OptimizeExtrinsics(
            r_world_to_cam,
            t_world_to_cam,
            key2d,
            key3d_op_smpl,
            intrinsics,
            height=height_m,
            smpl_model=smpl_model,
            beta=beta,
            static_cam=static_cam,
            iterations=10,
            printer=False,
        )
        output_opt1 = optimizer.optimize()

        # Safety check: if extrinsics optimization still produced NaN (e.g. degenerate
        # input geometry), fall back to the WHAM-derived initial values so that pose
        # optimization at least starts from a physically plausible camera pose.
        if not torch.isfinite(output_opt1["t"]).all() or not torch.isfinite(
            output_opt1["R"]
        ).all():
            logger.warning(
                "Extrinsics optimization returned NaN/Inf — falling back to "
                "initial WHAM-derived camera extrinsics for pose optimization."
            )
            output_opt1["t"] = t_world_to_cam[0]  # static-cam: single frame value
            output_opt1["R"] = r_world_to_cam[0]

        # Extract optimized camera parameters
        t_world_to_cam_opt1 = output_opt1["t"].repeat(n_frames, 1)
        r_world_to_cam_opt1 = output_opt1["R"].unsqueeze(0).repeat(n_frames, 1, 1)
        beta = output_opt1["beta"]

        # Project 3D keypoints to 2D using optimized camera parameters
        key2d_smpl_opt1 = ut.reproject(
            intrinsics, r_world_to_cam_opt1, t_world_to_cam_opt1, key3d_op_smpl
        )

        # Optimization 2: Root motion and pose parameters
        logger.info("Optimizing pose.")

        # Initialize and run pose optimization
        optimizer = OptimizePose(
            r_world_to_cam_opt1,
            t_world_to_cam_opt1,
            wham_result["pose_world"][:, :3],
            wham_result["trans_world"],
            intrinsics,
            key2d,
            smpl_model,
            wham_result["pose_world"][:, 3:],
            beta,
            wham_result["contact"],
            frame_rate,
            optimize_camera=optimize_camera,
            print_loss_terms=print_loss_terms,
            iterations=n_iter_opt2,
            weights=weights_opt2,
            cutoff_frequency=filter_freq,
            smoothness_diff_n=smoothness_diff_n,
            output_dir=output_path,  # Pass output path for visualization
            video_path=(
                trimmed_video_path
                if save_trimmed_video and trimmed_video_path is not None
                else video_path
            ),  # Pass video path for contact overlay
            frame_ids=frame_ids,  # Pass frame IDs for contact overlay
            create_contact_visualizations=create_contact_visualizations,  # Pass flag to control visualization creation
        )

        # Run optimization and update results
        output_opt2 = optimizer.optimize()

        # Extract optimized parameters
        t_root_in_world = output_opt2["t_root_in_world"]
        r_root_in_world = output_opt2["r_root_in_world"]
        t_world_to_cam = output_opt2["t_world_to_cam"].squeeze(0)
        r_world_to_cam = output_opt2["r_world_to_cam"].squeeze(0)
        body_pose = output_opt2["body_pose"]
        beta = output_opt2["beta"]

    except Exception as e:
        logger.error(f"Error in optimizer: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        if device == "cuda":
            torch.set_default_tensor_type("torch.FloatTensor")
        return output_paths

    # Calculate and save 3D keypoints and vertices in camera coordinates

    if save_smpl_for_viz:
        try:
            logger.info("Calculating 3D coordinates in camera frame...")
            key3d_world = output_opt2["key_3d"].squeeze(0)
            r_world_to_cam_opt = r_world_to_cam
            t_world_to_cam_opt = t_world_to_cam

            # Compute vertices if not already in output_opt2
            if "vertices" not in output_opt2 or output_opt2["vertices"] is None:
                logger.info("Computing vertices from SMPL parameters...")
                try:
                    # Get combined pose in world frame (root + body)
                    pose_world_combined = torch.cat(
                        [
                            r_root_in_world.reshape(len(r_root_in_world), -1)[
                                :, :3
                            ],  # Take first 3 components
                            body_pose.reshape(len(body_pose), -1),
                        ],
                        dim=1,
                    )

                    # Compute vertices for all frames
                    vertices_list = []
                    faces_np = None  # Will get faces from first smpl_result
                    for i_frame in range(len(t_root_in_world)):
                        smpl_result = ut.pred_smpl(
                            smpl_model,
                            trans=t_root_in_world[i_frame : i_frame + 1].reshape(
                                (1, 1, 3)
                            ),
                            root_orient=pose_world_combined[i_frame, :3].reshape(
                                (1, 1, 3)
                            ),
                            body_pose=pose_world_combined[i_frame, 3:].reshape(
                                (1, 1, -1)
                            ),
                            betas=beta.reshape((1, len(beta))),
                        )

                        # Check if vertices are available in smpl_result
                        if "verts3d_all" in smpl_result:
                            verts = smpl_result["verts3d_all"]
                            # Squeeze all batch dimensions to get (n_vertices, 3)
                            while verts.dim() > 2:
                                verts = verts.squeeze(0)
                            vertices_list.append(verts)  # Should be (n_vertices, 3)

                            # Get faces from first result (faces are static)
                            if faces_np is None and "faces" in smpl_result:
                                faces_np = smpl_result["faces"]
                                logger.info(
                                    f"Extracted SMPL faces from smpl_result, shape: {faces_np.shape}"
                                )
                        else:
                            logger.warning(
                                f"No verts3d_all in smpl_result at frame {i_frame}. Available keys: {list(smpl_result.keys())}"
                            )
                            break

                    if len(vertices_list) == len(t_root_in_world):
                        # Stack all vertices
                        vertices_world_all = torch.stack(
                            vertices_list, dim=0
                        )  # Shape: (n_frames, n_verts, 3)
                        output_opt2["vertices"] = vertices_world_all.unsqueeze(
                            0
                        )  # Add batch dim for consistency
                        logger.info(
                            f"Computed vertices with shape: {output_opt2['vertices'].shape}"
                        )

                        # Store faces if extracted
                        if faces_np is not None:
                            output_opt2["faces"] = faces_np
                            logger.info(
                                f"Stored SMPL faces in output_opt2, shape: {faces_np.shape}"
                            )
                    else:
                        logger.error("Failed to compute vertices for all frames")
                        output_opt2["vertices"] = None

                except Exception as e:
                    logger.error(f"Failed to compute vertices: {e}")
                    output_opt2["vertices"] = None

            # Transform keypoints to camera coordinates
            key3d_cam_transposed = torch.einsum(
                "...ij,...kj->...ik", r_world_to_cam_opt, key3d_world
            )
            key3d_cam = key3d_cam_transposed.transpose(
                -1, -2
            ) + t_world_to_cam_opt.unsqueeze(1)

            keypoints_3d_cam_path = os.path.join(
                output_path, f"{trial_name}_keypoints_3d_cam.pkl"
            )
            with open(keypoints_3d_cam_path, "wb") as f:
                pickle.dump(key3d_cam.cpu().numpy(), f)
            output_paths["keypoints_3d_cam_pkl"] = keypoints_3d_cam_path
            logger.info(
                f"Saved 3D keypoints in camera coordinates to {keypoints_3d_cam_path}"
            )

            if "vertices" in output_opt2 and output_opt2["vertices"] is not None:
                vertices_world = output_opt2["vertices"].squeeze(0)

                # Ensure vertices have correct shape (n_frames, n_vertices, 3)
                while vertices_world.dim() > 3:
                    vertices_world = vertices_world.squeeze(
                        1
                    )  # Remove extra dimensions

                logger.info(f"Processing vertices with shape: {vertices_world.shape}")

                # Transform vertices to camera coordinates
                vertices_cam_transposed = torch.einsum(
                    "...ij,...kj->...ik", r_world_to_cam_opt, vertices_world
                )
                vertices_cam = vertices_cam_transposed.transpose(
                    -1, -2
                ) + t_world_to_cam_opt.unsqueeze(1)

                vertices_3d_cam_path = os.path.join(
                    output_path, f"{trial_name}_vertices_3d_cam.pkl"
                )
                with open(vertices_3d_cam_path, "wb") as f:
                    pickle.dump(vertices_cam.cpu().numpy(), f)
                output_paths["vertices_3d_cam_pkl"] = vertices_3d_cam_path
                logger.info(
                    f"Saved 3D vertices in camera coordinates to {vertices_3d_cam_path}"
                )

        except Exception as e:
            logger.error(f"Error calculating or saving 3D camera coordinates: {e}")

    # Generate plots if requested
    if plotting:
        try:
            # Control whether to show plots
            show_plot = False  # Keep False for non-interactive runs

            # Plot the objective function
            plot_obj_path = plot_objective_function(
                output_opt2, show=show_plot, save_path=output_path
            )
            output_paths["plot_objective"] = plot_obj_path

            # Project 3D keypoints from optimization 2 to 2D
            key2d_smpl_opt2 = ut.reproject(
                intrinsics,
                r_world_to_cam,
                t_world_to_cam,
                output_opt2["key_3d"].squeeze(),
            )

            # Prepare data for 2D keypoint visualization
            key2d_plot = {
                "image": key2d.cpu().numpy(),
                "opt1": key2d_smpl_opt1.cpu().squeeze(0).numpy(),
                "opt2": key2d_smpl_opt2.detach().cpu().squeeze(0).numpy(),
            }

            # Plot 2D keypoints
            fig_2d_path = plot_2d_keypoints_interactive_plotly(
                key2d_plot,
                save_path=output_path,
                range_mono=opencap_mono_frame_range_video_ref,
            )
            output_paths["plot_2d"] = fig_2d_path
            if show_plot:
                # Open in browser
                webbrowser.open(fig_2d_path)

            # Prepare data for 3D keypoint visualization
            key3d_plot = {
                "wham_output": key3d_op_smpl.cpu().squeeze(0).numpy(),
                "opt2": output_opt2["key_3d"].cpu().numpy().squeeze(),
                "com": output_opt2["com"].cpu().unsqueeze(1).repeat(1, 25, 1).numpy(),
            }

            if show_plot:
                # Assuming plot_3d_keypoints_interactive_plotly exists and returns path
                # fig_3d_path = plot_3d_keypoints_interactive_plotly(key3d_plot, save_path=output_path)
                # output_paths['plot_3d'] = fig_3d_path
                # webbrowser.open(fig_3d_path)
                logger.info("Showing plots")
        except Exception as e:
            logger.error(f"Error plotting: {e}")

    # Run OpenSim on optimized results if requested
    try:
        if run_opensim_opt2:
            # Initialize SMPL to Opensim converter for optimized results
            smpl2osim_opt2 = utils_opensim.SMPL_to_Opensim(
                smpl_model,
                root_trans=t_root_in_world,
                root_orient=r_root_in_world,
                body_pose=body_pose,
                beta=beta,
                output_dir=output_path,
                trial_name=trial_name + "_" + case,
                frame_rate=frame_rate,
                mass=mass_kg,
            )

            # Write TRC file
            logger.info("Writing TRC")
            trc_path = smpl2osim_opt2.smpl_to_trc(rotations=trc_rot)
            output_paths["trc_file"] = trc_path

            # Scale Opensim Model
            logger.info("Scaling Model")
            scaled_model_path = smpl2osim_opt2.scale_model()
            output_paths["scaled_model_file"] = scaled_model_path

            # Run IK
            logger.info("Running IK")
            ik_path = smpl2osim_opt2.run_ik()
            output_paths["ik_results_file"] = ik_path

    except Exception as e:
        logger.error(f"Error in Opensim: {e}")

    if save_smpl_for_viz:
        # Save optimized results for visualization
        optimized_pkl_path = save_optimized_results_for_viz(
            output_opt2, beta, output_path, trial_name
        )
        output_paths["optimized_pkl"] = optimized_pkl_path

    output_paths["predicted_activity"] = predicted_activity
    output_paths["activity_detection_method"] = activity_detection_method

    if device == "cuda":
        torch.set_default_tensor_type("torch.FloatTensor")

    return output_paths


def save_optimized_results_for_viz(output_opt, beta, output_path, trial_name):
    """
    Save the optimized results in a format compatible with WHAM visualization.
    Includes pre-computed vertices, joints, faces, and other fields for fast visualization.

    Parameters:
    -----------
    output_opt : dict
        Dictionary containing optimization results
    beta : torch.Tensor
        SMPL shape parameters
    output_path : str
        Directory to save the results
    trial_name : str
        Name of the trial

    Returns:
    --------
    saved_path : str
        Path to the saved file
    """
    # Create directory if it doesn't exist
    logger.info(f"Saving optimized results to {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Extract relevant parameters
    t_root_in_world = output_opt["t_root_in_world"]
    r_root_in_world = output_opt["r_root_in_world"]
    body_pose = output_opt["body_pose"]
    vertices = output_opt["vertices"] if "vertices" in output_opt else None
    key_3d = output_opt["key_3d"] if "key_3d" in output_opt else None

    # Convert to numpy for saving
    t_root_in_world_np = t_root_in_world.detach().cpu().numpy()
    r_root_in_world_np = r_root_in_world.detach().cpu().numpy()
    body_pose_np = body_pose.detach().cpu().numpy()
    beta_np = beta.detach().cpu().numpy()

    # Handle vertices - pre-compute if not available
    if vertices is not None:
        vertices_np = (
            vertices.detach().cpu().numpy().squeeze()
        )  # Remove batch dimension if present
        logger.info(
            f"Using vertices from optimization output, shape: {vertices_np.shape}"
        )
    else:
        # Try to load vertices from WHAM output as fallback
        wham_output_path = os.path.join(output_path, "wham_output.pkl")
        if os.path.exists(wham_output_path):
            with open(wham_output_path, "rb") as f:
                wham_results = joblib.load(f)
            wham_subject_id = list(wham_results.keys())[0]
            vertices_np = wham_results[wham_subject_id]["verts"]
            logger.info(f"Using vertices from WHAM output, shape: {vertices_np.shape}")
        else:
            vertices_np = None
            logger.warning(
                "No vertices found - visualization will compute them on-the-fly"
            )

    # Handle 3D joints/keypoints
    if key_3d is not None:
        joints_np = (
            key_3d.detach().cpu().numpy().squeeze()
        )  # Remove batch dimension if present
        logger.info(f"Using joints from optimization output, shape: {joints_np.shape}")
    else:
        joints_np = None
        logger.warning("No pre-computed joints found")

    # Try to get SMPL faces for mesh rendering
    faces_np = None
    try:
        # Get faces from output_opt if available (extracted during vertex computation)
        if "faces" in output_opt:
            faces_np = output_opt["faces"]
            if hasattr(faces_np, "detach"):  # Convert from tensor if needed
                faces_np = faces_np.detach().cpu().numpy()
            logger.info(
                f"Using SMPL faces from optimization output, shape: {faces_np.shape}"
            )
        else:
            logger.info("No faces found in optimization output")

    except Exception as e:
        logger.warning(f"Could not extract faces from output_opt: {e}")

    # Create expanded betas array (per-frame for compatibility)
    n_frames = len(t_root_in_world_np)
    betas_expanded = np.tile(beta_np, (n_frames, 1))

    # Create pose array in axis-angle format (72 parameters per frame)
    pose_world_combined = np.concatenate(
        [
            r_root_in_world_np.reshape(
                n_frames, -1
            ),  # Root orientation (3 or 9 params)
            body_pose_np.reshape(n_frames, -1),  # Body pose (remaining params)
        ],
        axis=1,
    )

    # Ensure pose has exactly 72 parameters (standard SMPL format)
    if pose_world_combined.shape[1] != 72:
        if pose_world_combined.shape[1] > 72:
            pose_world_combined = pose_world_combined[:, :72]
            logger.warning(f"Truncated pose parameters to 72")
        else:
            # Pad with zeros if needed
            padding = 72 - pose_world_combined.shape[1]
            pose_world_combined = np.pad(
                pose_world_combined, ((0, 0), (0, padding)), "constant"
            )
            logger.warning(f"Padded pose parameters with {padding} zeros")

    # Create a comprehensive dictionary in WHAM format with additional fields
    wham_format_results = {
        "0": {
            # Core SMPL parameters
            "pose_world": pose_world_combined,  # (n_frames, 72) - axis-angle pose
            "pose": pose_world_combined,  # Alias for compatibility
            "trans_world": t_root_in_world_np,  # (n_frames, 3) - world translation
            "trans": t_root_in_world_np,  # Alias for compatibility
            "betas": betas_expanded,  # (n_frames, 10) - shape params per frame
            # Pre-computed geometry (for fast visualization)
            "verts": vertices_np,  # (n_frames, 6890, 3) - mesh vertices
            "joints": joints_np,  # (n_frames, N, 3) - 3D joints/keypoints
            # Mesh topology (static, for rendering)
            "faces": faces_np,  # (N_faces, 3) - mesh face indices
            # Frame information
            "frame_ids": np.arange(n_frames),  # (n_frames,) - frame indices
            # Additional metadata
            "n_frames": n_frames,
            "n_vertices": vertices_np.shape[1] if vertices_np is not None else 6890,
            "trial_name": trial_name,
            "optimization_type": "mono_optimized",
            # Camera info if available
            "cam_R": output_opt.get("r_world_to_cam", None),
            "cam_T": output_opt.get("t_world_to_cam", None),
        }
    }

    # Add camera parameters if available
    if "r_world_to_cam" in output_opt:
        cam_R = output_opt["r_world_to_cam"].detach().cpu().numpy()
        wham_format_results["0"]["cam_R"] = cam_R
        logger.info(f"Added camera rotation, shape: {cam_R.shape}")

    if "t_world_to_cam" in output_opt:
        cam_T = output_opt["t_world_to_cam"].detach().cpu().numpy()
        wham_format_results["0"]["cam_T"] = cam_T
        logger.info(f"Added camera translation, shape: {cam_T.shape}")

    # Save to file
    saved_path = os.path.join(output_path, f"{trial_name}_optimized.pkl")
    with open(saved_path, "wb") as f:
        joblib.dump(wham_format_results, f)  # Use joblib for consistency

    # Print summary
    logger.info(f"Saved optimized results for visualization to {saved_path}")
    logger.info(f"Summary:")
    logger.info(f"  - Frames: {n_frames}")
    logger.info(
        f"  - Vertices: {'✓' if vertices_np is not None else '✗'} {vertices_np.shape if vertices_np is not None else 'None'}"
    )
    logger.info(
        f"  - Joints: {'✓' if joints_np is not None else '✗'} {joints_np.shape if joints_np is not None else 'None'}"
    )
    logger.info(
        f"  - Faces: {'✓' if faces_np is not None else '✗'} {faces_np.shape if faces_np is not None else 'None'}"
    )
    logger.info(
        f"  - Camera params: {'✓' if 'cam_R' in wham_format_results['0'] else '✗'}"
    )

    return saved_path


# test = 1
