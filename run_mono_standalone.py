#!/usr/bin/env python3
"""
Standalone script to run the mono pipeline without the API.
This is useful for debugging with breakpoints.

Usage:
    python run_mono_standalone.py
"""

import os
import time
import yaml
import hashlib
from loguru import logger
from hashlib import md5
from pathlib import Path
from typing import Optional

# Import the same modules used in mono_api.py
from optimization import run_optimization
from WHAM.demo import main_wham
from visualization.utils import generateVisualizerJson
from visualization.automation import automate_recording
from utils.convert_to_avi import convert_to_avi
from utils.utilsCameraPy3 import getVideoRotation
from utils.tracking_filters import InsufficientFullBodyKeypointsError

# Import enhanced logging (optional)
try:
    from deployment.logging_config import setup_logging

    setup_logging("api")
except ImportError:
    logger.warning("Enhanced logging not available, using basic logging")

# Get repo path
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def generate_request_hash(video_path, metadata):
    """Generate a unique hash to identify a processing request based on its parameters."""
    # Create a string with all relevant parameters
    # Use 'sex' key consistently, providing defaults if missing
    sex = metadata.get("sex", "unknown")
    height = metadata.get("height_m", 0.0)
    mass = metadata.get("mass_kg", 0.0)
    key_string = f"{video_path}_{height}_{mass}_{sex}"
    # Generate a hash
    return hashlib.md5(key_string.encode()).hexdigest()


def resolve_intrinsics_from_metadata(metadata: dict, repo_path: str) -> str:
    """Resolve device-specific intrinsics from metadata (same logic as mono_api)."""
    device_model = None
    try:
        device_model = metadata.get("iphoneModel", {}).get("Cam0", "")
        device_model = device_model.replace("iphone", "iPhone").replace(
            "ipad", "iPad"
        )
    except Exception:
        pass

    default_intrinsics = os.path.join(
        repo_path, "examples/Intrinsics/iphone12Pro_intrinsics.pickle"
    )
    if device_model:
        device_intrinsics = os.path.join(
            repo_path,
            f"camera_intrinsics/{device_model}/Deployed/cameraIntrinsics.pickle",
        )
        if os.path.exists(device_intrinsics):
            logger.info(
                f"Using device-specific intrinsics for {device_model}: {device_intrinsics}"
            )
            return device_intrinsics
        logger.warning(
            f"No intrinsics found for device '{device_model}' at {device_intrinsics}. "
            f"Falling back to {default_intrinsics}"
        )
        return default_intrinsics

    logger.warning(
        f"Could not determine device model from metadata. "
        f"Falling back to {default_intrinsics}"
    )
    return default_intrinsics


def run_mono_standalone(
    video_path: str,
    metadata_path: str,
    calib_path: str,
    intrinsics_path: Optional[str] = None,
    estimate_local_only: bool = True,
    rerun: bool = False,
    session_id: Optional[str] = None,
    activity: Optional[str] = None,
):
    """
    Run the mono pipeline standalone (without API).

    Args:
        video_path: Path to the input video file
        metadata_path: Path to the metadata YAML file
        calib_path: Path to the calibration file
        intrinsics_path: Path to intrinsics pickle; if None or empty, resolved from
            metadata iphoneModel.Cam0 like the API (camera_intrinsics/.../cameraIntrinsics.pickle).
        estimate_local_only: Whether to estimate local only
        rerun: Whether to rerun even if cached results exist
        session_id: Optional session ID to use as case_id
        activity: Optional activity type (e.g., "walking", "sitting")

    Returns:
        Dictionary with results similar to the API response
    """
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    if not intrinsics_path:
        intrinsics_path = resolve_intrinsics_from_metadata(metadata, repo_path)

    height_m = metadata.get("height_m", 1.70)  # Default height if not found
    mass_kg = metadata.get("mass_kg", 70.0)  # Default mass if not found
    sex = metadata.get("sex", "male")  # Default sex if not found
    logger.info(f"Height: {height_m} m, Mass: {mass_kg} kg, Sex: {sex}")

    # If rerun is False, check for cached results
    if not rerun:
        request_hash = generate_request_hash(video_path, metadata)
        results_dir = os.path.join(repo_path, "results")

        if os.path.exists(results_dir):
            for case_dir in os.listdir(results_dir):
                cache_file = os.path.join(results_dir, case_dir, "request_hash.txt")
                if os.path.exists(cache_file):
                    with open(cache_file, "r") as f:
                        stored_hash = f.read().strip()

                    if stored_hash == request_hash:
                        logger.info(f"Found cached results in {case_dir}")
                        video_name = os.path.basename(video_path).split(".")[0]
                        results_path = os.path.join(results_dir, case_dir, video_name)

                        # Check if key files exist
                        ik_file_path = os.path.join(results_path, "ik_results.pkl")
                        output_mono_json_path = os.path.join(results_path, "mono.json")
                        output_video_path = os.path.join(
                            results_path, "viewer_mono.webm"
                        )

                        if os.path.exists(ik_file_path) and os.path.exists(
                            output_mono_json_path
                        ):
                            return {
                                "message": "Mono pipeline completed successfully (cached result)!",
                                "ik_file_path": ik_file_path,
                                "case_id": case_dir,
                                "case_dir": os.path.join(results_dir, case_dir),
                                "visualization": {
                                    "created": True,
                                    "json_path": output_mono_json_path,
                                    "video_path": (
                                        output_video_path
                                        if os.path.exists(output_video_path)
                                        else None
                                    ),
                                },
                            }

    # Create a case directory based on timestamp and request parameters
    if session_id:
        case_num = session_id
        logger.info(f"Using provided session_id as case_id: {case_num}")
    else:
        timestamp = int(time.time())
        # Hash includes parameters that might affect processing but aren't part of the input 'identity' for caching
        case_hash = md5(f"{timestamp}_{estimate_local_only}".encode()).hexdigest()[
            :8
        ]  # rerun is handled by the cache check logic
        case_num = f"{timestamp}_{case_hash}"
        logger.info(f"Generated new case_id: {case_num}")

    # Create case directory structure
    case_dir = os.path.join(repo_path, "results", case_num)
    os.makedirs(case_dir, exist_ok=True)

    logger.info(f"Created case directory: {case_dir}")

    # Create output directory
    video_name = os.path.basename(video_path).split(".")[0]  # Remove file extension
    # trial_path = os.path.join(case_dir, video_name)
    trial_path = case_dir
    logger.info(f"trial_path: {trial_path}")
    if not os.path.exists(trial_path):
        os.makedirs(trial_path)

    start_time = time.time()

    logger.info(f"video_path: {video_path}")
    logger.info(f"metadata_path: {metadata_path}")
    logger.info(f"calib_path: {calib_path}")
    logger.info(f"intrinsics_path: {intrinsics_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    # Determine video rotation
    rotation = getVideoRotation(video_path)
    logger.info(f"Rotation: {rotation}")

    # Convert MOV to AVI
    if video_path.lower().endswith(".mov"):
        logger.info(f"Converting MOV file to AVI: {video_path}")
        video_path = convert_to_avi(video_path)
        logger.info(f"Conversion complete. New video path: {video_path}")

    # Run WHAM (match flow: estimate_local_only for consistent world/cam coordinates)
    inputs_wham = {
        "calib_path": calib_path,
        "video_path": video_path,
        "output_path": trial_path,
        "visualize": True,
        "save_pkl": True,
        "run_smplify": True,
        "rerun": rerun,
        "estimate_local_only": estimate_local_only,
    }

    logger.info(
        "Starting WHAM: watch [WHAM preprocess] and [keypoint_filter] logs for "
        "frame counts and 2D keypoint confidence."
    )
    try:
        main_wham(**inputs_wham)
    except InsufficientFullBodyKeypointsError as e:
        logger.warning(f"Mono pipeline stopped after WHAM keypoint gate: {e}")
        return {
            "message": "Aborted: video does not show a full body with sufficient 2D keypoint confidence.",
            "error": str(e),
            "aborted": True,
            "aborted_stage": "WHAM_preprocess",
            "case_id": case_num,
            "case_dir": case_dir,
            "metadata_path": metadata_path,
            "visualization": {"created": False, "json_path": None, "video_path": None},
        }
    logger.info("Wham done")
    logger.info(f"Time taken for Wham: {time.time() - start_time:.2f} seconds")

    # extract video name from video path. it's the last part of the path and without the extension
    video_name = os.path.basename(video_path).split(".")[0]

    # Run optimization
    results_path = (
        trial_path  # Use trial_path directly instead of joining with video basename
    )
    logger.info(f"results_path: {results_path}")

    # the result path is result path + the video name
    results_path = os.path.join(results_path, video_name)

    # Run the optimization
    logger.info("Running optimization...")

    inputs_optimization = {
        "data_dir": results_path,
        "trial_name": os.path.basename(
            results_path
        ),  # Use folder name instead of video filename
        "height_m": height_m,
        "mass_kg": mass_kg,
        "sex": sex,
        "intrinsics_pth": intrinsics_path,
        "run_opensim_original_wham": True,
        "run_opensim_opt2": True,
        "use_gpu": True,
        "static_cam": True,  # Static camera (fixed in optimization.py); use False for moving camera
        "n_iter_opt2": 75,
        "print_loss_terms": False,
        "plotting": True,
        "save_video_debug": False,
        "output_path": results_path,
        "video_path": video_path,
        "activity": activity,
        "rotation": rotation,
        "create_contact_visualizations": False,
    }

    output_paths = run_optimization(**inputs_optimization)

    logger.info("Optimization done")
    logger.info(f"Time taken for optimization: {time.time() - start_time:.2f} seconds")

    logger.info(f"output_paths: {output_paths}")

    # Generate visualization
    logger.info("Generating visualization...")

    # Get paths for visualization
    model_mono_sub_folder = os.path.join(results_path, "OpenSim", "Model")
    # Check if Model folder exists
    if not os.path.exists(model_mono_sub_folder):
        logger.error(f"Model folder does not exist: {model_mono_sub_folder}")
        model_mono_file = None
        ik_motion_file = None
    else:
        # find the folder which does not contain 'wham' in the name
        model_folders = [
            x for x in os.listdir(model_mono_sub_folder) if "wham" not in x
        ]
        if not model_folders:
            logger.error(
                f"No model folder found (excluding 'wham') in {model_mono_sub_folder}"
            )
            model_mono_file = None
            ik_motion_file = None
        else:
            model_mono_folder = os.path.join(
                results_path,
                "OpenSim",
                "Model",
                model_folders[0],
            )
            model_mono_file = os.path.join(
                model_mono_folder, "LaiUhlrich2022_scaled_no_patella.osim"
            )

            # Check if model file exists
            if not os.path.exists(model_mono_file):
                logger.warning(f"Model file not found: {model_mono_file}")
                model_mono_file = None

            # Get IK motion file
            ik_motion_sub_folder = os.path.join(results_path, "OpenSim", "IK")
            if not os.path.exists(ik_motion_sub_folder):
                logger.error(f"IK folder does not exist: {ik_motion_sub_folder}")
                ik_motion_file = None
            else:
                # find the folder which does not contain 'wham' in the name in ik_motion_sub_folder
                ik_folders = [
                    x for x in os.listdir(ik_motion_sub_folder) if "wham" not in x
                ]
                if not ik_folders:
                    logger.error(
                        f"No IK folder found (excluding 'wham') in {ik_motion_sub_folder}"
                    )
                    ik_motion_file = None
                else:
                    ik_motion_folder = os.path.join(
                        results_path,
                        "OpenSim",
                        "IK",
                        ik_folders[0],
                    )
                    # Check if IK folder exists and find .mot file
                    if not os.path.exists(ik_motion_folder):
                        logger.error(
                            f"IK motion folder does not exist: {ik_motion_folder}"
                        )
                        ik_motion_file = None
                    else:
                        mot_files = [
                            x
                            for x in os.listdir(ik_motion_folder)
                            if x.endswith(".mot")
                        ]
                        if not mot_files:
                            logger.error(
                                f"IK failed: No .mot file found in {ik_motion_folder}. "
                                "This usually means IK crashed (check for 'Floating point exception' in logs)."
                            )
                            logger.error(
                                f"Files in IK folder: {os.listdir(ik_motion_folder)}"
                            )
                            ik_motion_file = None
                        else:
                            ik_motion_file = os.path.join(
                                ik_motion_folder, mot_files[0]
                            )

    output_mono_json_path = os.path.join(results_path, "mono.json")
    output_video_path = os.path.join(results_path, "viewer_mono.webm")

    logger.info(f"model_mono_file: {model_mono_file}")
    logger.info(f"ik_motion_file: {ik_motion_file}")
    logger.info(f"output_mono_json_path: {output_mono_json_path}")
    logger.info(f"output_video_path: {output_video_path}")

    # Generate JSON for visualization
    jsonOutputPath = None
    if model_mono_file and ik_motion_file:
        try:
            jsonOutputPath = generateVisualizerJson(
                modelPath=model_mono_file,
                ikPath=ik_motion_file,
                jsonOutputPath=output_mono_json_path,
                vertical_offset=0,
            )
            logger.info(f"Generated visualization JSON file at {jsonOutputPath}")

            viz = False
            if viz:
                # Create visualization video
                automate_recording(
                    json_paths=[output_mono_json_path],
                    output_video_path=output_video_path,
                    num_loops=1,
                )
                logger.info("Generated visualization video")

            visualization_created = True
        except Exception as e:
            logger.error(f"Error in visualization generation: {str(e)}")
            visualization_created = False
    else:
        logger.warning(
            "Skipping visualization generation: IK failed or required files missing"
        )
        visualization_created = False

    ik_file_path = os.path.join(results_path, "ik_results.pkl")

    # Store the request hash before returning
    request_hash = generate_request_hash(video_path, metadata)
    hash_file_path = os.path.join(case_dir, "request_hash.txt")
    with open(hash_file_path, "w") as f:
        f.write(request_hash)
    logger.info(f"Stored request hash {request_hash} for future reference")

    response = {
        "message": "Mono pipeline completed successfully!",
        "ik_file_path": output_paths.get("ik_results_file"),
        "json_file_path": jsonOutputPath,
        "video_file_path": output_paths.get("trimmed_video"),
        "trc_file_path": output_paths.get("trc_file"),
        "scaled_model_file_path": output_paths.get("scaled_model_file"),
        "case_id": case_num,
        "case_dir": case_dir,
        "metadata_path": metadata_path,
        "pose_pickle_path": output_paths.get("optimized_pkl"),
        "keypoints_3d_cam_path": output_paths.get("keypoints_3d_cam_pkl"),
        "vertices_3d_cam_path": output_paths.get("vertices_3d_cam_pkl"),
        "visualization": {
            "created": visualization_created,
            "json_path": output_mono_json_path if visualization_created else None,
            "video_path": (
                output_video_path
                if visualization_created and os.path.exists(output_video_path)
                else None
            ),
        },
    }

    return response


if __name__ == "__main__":
    # Example usage - modify these paths as needed
    # You can set breakpoints anywhere in the run_mono_standalone function

    video_path = "tr1.mov"
    metadata_path = "tr1.yaml"
    calib_path = "calib.txt"
    # intrinsics_path: omit or None to resolve from metadata iphoneModel.Cam0 (mono_api behavior)

    # Optional: Initialize WHAM model (similar to API startup)
    try:
        from WHAM.demo import initialize_wham

        logger.info("Loading WHAM model...")
        initialize_wham()
        logger.info("WHAM model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not initialize WHAM model: {e}")

    # Run the pipeline
    result = run_mono_standalone(
        video_path=video_path,
        metadata_path=metadata_path,
        calib_path=calib_path,
        estimate_local_only=True,
        rerun=False,
        session_id="treadmill_dev",
        activity="treadmill",
    )

    # Print results
    logger.info("Pipeline completed!")
    logger.info(f"Results: {result}")
