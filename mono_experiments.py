import os
import yaml
from loguru import logger
from optimization import run_optimization
import torch
import time
from WHAM.demo import main_wham
from utils.utilsCameraPy3 import getVideoRotation
from utils.convert_to_avi import convert_to_avi

torch.cuda.empty_cache()

repo_path = os.path.dirname(os.path.abspath(__file__))
validation_videos_path = os.path.join(repo_path, "LabValidation_withVideos1")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

loop = False

movements = ["walking", "squats", "STS"]
cameras = ["Cam1"]
range_subjects = range(5, 6)

# Load parameters from YAML file
with open("params/parameters.yaml", "r") as f:
    params = yaml.safe_load(f)

filter_freq = params["filter_freq"]
weights_opt2_dj = params["weights_opt2_dj"]
weights_opt2_walking = params["weights_opt2_walking"]
weights_opt2_squats = params["weights_opt2_squats"]
weights_opt2_sts = params["weights_opt2_sts"]

if loop:
    for i in range_subjects:
        subject = f"subject{i}"
        metadata = os.path.join(validation_videos_path, subject, "sessionMetadata.yaml")
        with open(metadata, "rb") as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)

        height_m = metadata["height_m"]
        mass_kg = metadata["mass_kg"]
        sex = metadata["sex"]
        logger.info(f"Height: {height_m} m, Mass: {mass_kg} kg, Sex: {sex}")
        start_time = time.time()

        videos_dir = os.path.join(validation_videos_path, subject, "VideoData")
        sessions = [
            f
            for f in os.listdir(videos_dir)
            if os.path.isdir(os.path.join(videos_dir, f))
        ]
        for session in sessions:
            session_path = os.path.join(videos_dir, str(session))
            cams = [
                f
                for f in os.listdir(session_path)
                if os.path.isdir(os.path.join(session_path, f))
            ]
            cams = [cam for cam in cams if cam in cameras]
            for cam in cams:
                cam_path = os.path.join(session_path, cam)
                videos = [
                    f
                    for f in os.listdir(cam_path)
                    if os.path.isdir(os.path.join(cam_path, f))
                ]
                for video in videos:
                    video_path = os.path.join(cam_path, video)
                    files = [
                        f
                        for f in os.listdir(video_path)
                        if os.path.isfile(os.path.join(video_path, f))
                    ]
                    valid_video_extensions = [".mp4", ".avi"]
                    files = [
                        f
                        for f in files
                        if any(f.endswith(ext) for ext in valid_video_extensions)
                    ]
                    to_exclude = ["extrinsics", "syncdWithMocap"]
                    files = [
                        f
                        for f in files
                        if not any(exclude in f for exclude in to_exclude)
                    ]
                    for file in files:
                        if "_trimmed" in file:
                            files = [file]
                            break

                    for file in files:
                        if any(movement in file for movement in movements):
                            movement = [
                                movement for movement in movements if movement in file
                            ][0]
                            torch.cuda.synchronize()
                            logger.info(
                                f"Processing {subject} - {session} - {cam} - {video} - {file} ({movement}) ..."
                            )
                            step_start_time = time.time()

                            torch.cuda.empty_cache()

                            # Get full video path and convert MOV to AVI if necessary
                            video_path_full = os.path.join(video_path, file)
                            if video_path_full.lower().endswith(".mov"):
                                logger.info(f"Converting MOV file to AVI: {video_path_full}")
                                video_path_full = convert_to_avi(video_path_full)
                                logger.info(f"Conversion complete. New video path: {video_path_full}")

                            inputs_wham = {
                                "calib_path": "examples/walking4/calib.txt",
                                "video_path": video_path_full,
                                "output_path": os.path.join(
                                    repo_path, "output", subject, session, cam, video
                                ),
                                "visualize": True,
                                "estimate_local_only": True,
                                "save_pkl": True,
                                "run_smplify": True,
                            }

                            results = main_wham(**inputs_wham)
                            logger.info("Wham done")
                            logger.info(
                                f"Time taken for Wham: {time.time() - step_start_time:.2f} seconds"
                            )
                            step_start_time = time.time()

                            logger.info("Running optimization...")
                            video_name_trimmed = video
                            if "_trimmed" in file:
                                video_name_trimmed = file.split(".avi")[0]

                            results_path = os.path.join(
                                repo_path,
                                "output",
                                subject,
                                session,
                                cam,
                                video,
                                video_name_trimmed,
                            )
                            
                            # Determine video rotation
                            video_path_full = os.path.join(video_path, file)
                            rotation = getVideoRotation(video_path_full)
                            logger.info(f"Rotation: {rotation}")
                            
                            inputs_optimization = {
                                "data_dir": results_path,
                                "trial_name": video,
                                "height_m": height_m,
                                "mass_kg": mass_kg,
                                "sex": sex,
                                "intrinsics_pth": "examples/Intrinsics/iphone12Pro_intrinsics.pickle",
                                "weights_opt2": (
                                    weights_opt2_walking
                                    if movement == "walking"
                                    else (
                                        weights_opt2_squats
                                        if movement == "squats"
                                        else (
                                            weights_opt2_sts
                                            if movement == "STS"
                                            else weights_opt2_dj
                                        )
                                    )
                                ),
                                "case": "5",
                                "trc_rot": {"x": 0, "y": 46.5, "z": 0},
                                "run_opensim_original_wham": True,
                                "run_opensim_opt2": True,
                                "use_gpu": True,
                                "filter_freq": filter_freq[movement],
                                "static_cam": False,
                                "n_iter_opt2": 75,
                                "print_loss_terms": False,
                                "plotting": True,
                                "rotation": rotation,
                            }

                            results_optimization = run_optimization(
                                **inputs_optimization
                            )
                            logger.info("Optimization done")
                            logger.info(
                                f"Time taken for optimization: {time.time() - step_start_time:.2f} seconds"
                            )
                            logger.info(
                                f"Total time taken for {subject} - {session} - {cam} - {video} - {file}: {time.time() - start_time:.2f} seconds"
                            )
                            print("-" * 50)

    logger.info("ALL DONE !")

else:
    subject = "subject5"
    session = "Session1"
    cam = "Cam1"
    video = "walking3"
    video_name = "walking3.avi"
    movement = "walking"

    run_wham = True

    subject_path = os.path.join(validation_videos_path, subject)
    folder_path = os.path.join(subject_path, "VideoData", session, cam, video)

    start_time = time.time()

    torch.cuda.empty_cache()

    # Get full video path and convert MOV to AVI if necessary
    video_path_full = os.path.join(folder_path, video_name)
    if video_path_full.lower().endswith(".mov"):
        logger.info(f"Converting MOV file to AVI: {video_path_full}")
        video_path_full = convert_to_avi(video_path_full)
        logger.info(f"Conversion complete. New video path: {video_path_full}")

    inputs_wham = {
        "calib_path": "examples/walking4/calib.txt",
        "video_path": video_path_full,
        "output_path": os.path.join(repo_path, "output", subject, session, cam, video),
        "visualize": True,
        "estimate_local_only": True,
        "save_pkl": True,
        "run_smplify": True,
    }

    if run_wham:
        results = main_wham(**inputs_wham)
        logger.info("Wham done")
    else:
        logger.info("Skipping Wham")
        logger.info(f"Time taken for Wham: {time.time() - start_time:.2f} seconds")

    metadata = os.path.join(subject_path, "sessionMetadata.yaml")
    with open(metadata, "rb") as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)

    height_m = metadata["height_m"]
    mass_kg = metadata["mass_kg"]
    sex = metadata["sex"]
    logger.info(f"Height: {height_m} m, Mass: {mass_kg} kg, Sex: {sex}")
    video_name_trimmed = video_name.split(".avi")[0]
    if "_trimmed" in video_name:
        video_name_trimmed = video_name.split(".avi")[0]

    results_path = os.path.join(
        repo_path, "output", subject, session, cam, video, video_name_trimmed
    )
    step_start_time = time.time()
    
    # Determine video rotation
    video_path_full = os.path.join(folder_path, video_name)
    rotation = getVideoRotation(video_path_full)
    logger.info(f"Rotation: {rotation}")
    
    inputs_optimization = {
        "data_dir": results_path,
        "trial_name": video,
        "height_m": height_m,
        "mass_kg": mass_kg,
        "sex": sex,
        "intrinsics_pth": "examples/Intrinsics/iphone12Pro_intrinsics.pickle",
        "weights_opt2": (
            weights_opt2_walking
            if movement == "walking"
            else (
                weights_opt2_squats
                if movement == "squats"
                else weights_opt2_sts if movement == "STS" else weights_opt2_dj
            )
        ),
        "case": "5",
        "trc_rot": {"x": 0, "y": 46.5, "z": 0},
        "run_opensim_original_wham": True,
        "run_opensim_opt2": True,
        "use_gpu": True,
        "filter_freq": filter_freq[movement],
        "static_cam": False,
        "optimize_camera": True,
        "n_iter_opt2": 75,
        "print_loss_terms": False,
        "smoothness_diff_n": 2,
        "plotting": True,
        "rotation": rotation,
    }

    results_optimization = run_optimization(**inputs_optimization)
    logger.info("Optimization done")
    logger.info(
        f"Time taken for optimization: {time.time() - step_start_time:.2f} seconds"
    )
    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
