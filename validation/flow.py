import os
import yaml
import sys
import time
from loguru import logger
from time_sync import run_time_sync
from space_sync import run_space_sync_and_ik
from ik_analysis import run_ik_analysis
import torch
from visualization.utils import generateVisualizerJson
from visualization.automation import automate_recording
from marker_analysis import run_marker_analysis


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimization import run_optimization
from WHAM.demo import main_wham
from utils.utilsCameraPy3 import getVideoRotation
from utils.convert_to_avi import convert_to_avi

# Add NAS sync utilities
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_path)

# Import NAS sync function - we'll use this later
try:
    from nas.sync import main as sync_to_nas

    nas_sync_available = True
except ImportError:
    logger.warning("NAS sync module not available. Files will be saved locally only.")
    nas_sync_available = False

os.chdir(repo_path)
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# def flow(subject='subject4', session='Session1', cam='Cam3', video='walking1', video_name='walking1_trimmed.avi', activity='walk',
#          case_num="case_fixed_112", rerun=True):
#
def flow(
    subject="subject11",
    session="Session1",
    cam="Cam3",
    video="walking2",
    video_name="walking2_trimmed.avi",
    activity="walking",
    case_num="01_case_walking",
    rerun=True,
):
    global repo_path
    validation_videos_path = os.path.join(repo_path, "LabValidation_withVideos1")

    # create folder case_num if it doesn't exist
    case_path = os.path.join(
        repo_path, "output", case_num, subject, session, cam, video
    )
    if not os.path.exists(case_path):
        os.makedirs(case_path)

    # if the files are already in the case_path, skip the rest of the code
    if os.path.exists(
        os.path.join(case_path, "OpenSim", "IK", "shiftedIK", "translation_error.txt")
    ):
        print("Files already in the case path. Skipping...")

    subject_path = os.path.join(validation_videos_path, subject)
    session_path = os.path.join(subject_path, "VideoData", session)
    cam_path = os.path.join(session_path, cam)
    video_path = os.path.join(cam_path, video)

    with open("params/parameters.yaml", "r") as f:
        params = yaml.safe_load(f)

    # log the weights, filter freq, and smoothness diff n in the folder case_num
    base_path = os.path.join(repo_path, "output", case_num)
    # with open(os.path.join(base_path, "parameters.yaml"), "w") as f:
    # yaml.dump({"weights": weights, "filter_freq": filter_freq, "smoothness_diff_n": smoothness_diff_n}, f)

    output_path = os.path.join(repo_path, "output")
    # check if there is a file called 'flow_cases.yaml' in the output path
    if os.path.exists(os.path.join(output_path, "flow_cases.yaml")):
        with open(os.path.join(output_path, "flow_cases.yaml"), "r") as f:
            flow_cases = yaml.safe_load(f)
    else:
        flow_cases = []

    # save the flow_cases list to the output path
    with open(os.path.join(output_path, "flow_cases.yaml"), "w") as f:
        yaml.dump(flow_cases, f)

    start_time = time.time()

    # Always use local output path first, then sync to NAS
    use_nas = True
    local_output_path = os.path.join(repo_path, "output", subject, session, cam, video)

    # Create the local output directory
    os.makedirs(local_output_path, exist_ok=True)

    # Get full video path and convert MOV to AVI if necessary
    video_path_full = os.path.join(video_path, video_name)
    if video_path_full.lower().endswith(".mov"):
        logger.info(f"Converting MOV file to AVI: {video_path_full}")
        video_path_full = convert_to_avi(video_path_full)
        logger.info(f"Conversion complete. New video path: {video_path_full}")

    inputs_wham = {
        "calib_path": "examples/walking4/calib.txt",
        "video_path": video_path_full,
        "output_path": local_output_path,
        "visualize": False,
        "estimate_local_only": True,
        "save_pkl": True,
        "run_smplify": True,
        "rerun": False,
    }

    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary())

    results = main_wham(**inputs_wham)
    logger.info("Wham done")
    logger.info(f"Time taken for Wham: {time.time() - start_time:.2f} seconds")

    torch.cuda.empty_cache()

    metadata = os.path.join(subject_path, "sessionMetadata.yaml")
    with open(metadata, "rb") as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)

    height_m = metadata["height_m"] + 0.015
    mass_kg = metadata["mass_kg"]
    sex = metadata["sex"]
    logger.info(f"Height: {height_m} m, Mass: {mass_kg} kg, Sex: {sex}")
    video_name_trimmed = video_name.split(".avi")[0]

    # Use local paths for results
    results_path = os.path.join(
        repo_path, "output", subject, session, cam, video, video_name_trimmed
    )
    step_start_time = time.time()

    # empty gpu
    torch.cuda.empty_cache()

    # Determine video rotation
    video_path_full = os.path.join(video_path, video_name)
    rotation = getVideoRotation(video_path_full)
    logger.info(f"Rotation: {rotation}")

    inputs_optimization = {
        "data_dir": results_path,
        "trial_name": video,
        "height_m": height_m,
        "mass_kg": mass_kg,
        "sex": sex,
        "intrinsics_pth": "examples/Intrinsics/iphone12Pro_intrinsics.pickle",
        "case": "5",
        "run_opensim_original_wham": True,
        "run_opensim_opt2": True,
        "use_gpu": True,
        "save_video_debug": False,
        "static_cam": False,
        "print_loss_terms": False,
        "plotting": True,
        "output_path": case_path,
        "video_path": video_path_full,
        "activity": activity,
        "rotation": rotation,
    }

    if rerun:
        output_paths = run_optimization(**inputs_optimization)
    else:
        output_paths = None

    logger.info("Optimization done")
    logger.info(
        f"Time taken for optimization: {time.time() - step_start_time:.2f} seconds"
    )

    if output_paths:
        if output_paths.get("keypoints_3d_cam_pkl"):
            logger.info(
                f"3D keypoints in camera space path: {output_paths.get('keypoints_3d_cam_pkl')}"
            )
        if output_paths.get("vertices_3d_cam_pkl"):
            logger.info(
                f"3D vertices in camera space path: {output_paths.get('vertices_3d_cam_pkl')}"
            )

    lag, max_corr, graph_path = run_time_sync(
        subject,
        session,
        cam,
        movement=video,
        output_case_path=case_path,
        visualize=True,
    )
    # logger.info(f"Lag: {lag}, Max Correlation: {max_corr}")

    # let's assume we can use the same lag for wham than the one we got from the mono time sync
    marker_wham_path = None
    marker_wham_path_folder = None
    marker_wham_path_sub_folder = None

    folders = os.listdir(local_output_path)
    if len(folders) == 1:
        marker_wham_path_folder = os.path.join(
            local_output_path, folders[0], "MarkerData"
        )
    else:
        for folder in folders:
            if "trimmed" in folder:
                marker_wham_path_folder = os.path.join(
                    local_output_path, folder, "MarkerData"
                )
                break

    for folder in os.listdir(marker_wham_path_folder):
        if "wham_result" in folder:
            marker_wham_path_sub_folder = os.path.join(marker_wham_path_folder, folder)
            break

    for file in os.listdir(marker_wham_path_sub_folder):
        if "wham_result.trc" in file:
            marker_wham_path = os.path.join(marker_wham_path_sub_folder, file)
            break

    logger.info(f"Marker wham path: {marker_wham_path}")

    movement_path, synced_path, pathOutputMotion = run_space_sync_and_ik(
        subject,
        session,
        cam,
        movement=video,
        marker_wham_path=marker_wham_path,
        output_case_path=case_path,
    )
    logger.info("Space sync and IK analysis done")

    # ik_results = run_ik_analysis(subject, session, cam=, movement=video)

    video_is_trimmed = False
    if "trimmed" in video_name:
        video_is_trimmed = True

    (
        ik_results_degrees,
        ik_results_mm,
        ik_results_degrees_wham,
        ik_results_mm_wham,
        ik_results_degrees_2cams,
        ik_results_mm_2cams,
        results_path,
        output_csv_path,
    ) = run_ik_analysis(
        subject,
        session,
        cam,
        video,
        trimmed=video_is_trimmed,
        output_case_path=case_path,
        run_wham=True,
        run_2cams=True,
    )

    # Print results for all methods
    print(f"Mono - MAE degrees: {ik_results_degrees}, MAE mm: {ik_results_mm}")
    print(
        f"WHAM - MAE degrees: {ik_results_degrees_wham}, MAE mm: {ik_results_mm_wham}"
    )
    print(
        f"2CAMS - MAE degrees: {ik_results_degrees_2cams}, MAE mm: {ik_results_mm_2cams}"
    )
    logger.info("IK analysis done")

    marker_results = run_marker_analysis(
        subject,
        session,
        cam,
        video,
        exisiting_results_path_csv=output_csv_path,
        output_path_csv=output_csv_path,
        trimmed=video_is_trimmed,
        output_case_path=case_path,
        run_wham=True,
        run_2cams=False,
    )

    # Print marker analysis results
    print(f"Mono - Marker MAE: {marker_results['mono']} mm")
    if "wham" in marker_results:
        print(f"WHAM - Marker MAE: {marker_results['wham']} mm")
    if "2cams" in marker_results:
        print(f"2CAMS - Marker MAE: {marker_results['2cams']} mm")
    logger.info("Marker analysis done")

    # Mocap
    mocap_folder_path = os.path.join(movement_path, "mocap")
    # find a .osim in this folder
    mocap_model_file = None
    for file in os.listdir(mocap_folder_path):
        if file.endswith(".osim"):
            mocap_model_file = os.path.join(mocap_folder_path, file)
            break
    mocap_if_file = None
    for file in os.listdir(mocap_folder_path):
        if file.endswith(".mot"):
            mocap_if_file = os.path.join(mocap_folder_path, file)
            break
    assert mocap_model_file is not None, "No .osim file found in the mocap folder"
    assert mocap_if_file is not None, "No .mot file found in the mocap folder"

    output_mocap_json_path = os.path.join(mocap_folder_path, "mocap.json")

    generateVisualizerJson(
        modelPath=mocap_model_file,
        ikPath=mocap_if_file,
        jsonOutputPath=output_mocap_json_path,
        vertical_offset=0,
    )

    assert os.path.exists(output_mocap_json_path), "Mocap json file not created"

    # Mono
    output_mono_json_path = os.path.join(
        "/", *pathOutputMotion.split("/")[:-1], "mono.json"
    )
    model_mono_file = os.path.join(
        pathOutputMotion.split("IK")[0],
        "Model",
        synced_path.split("/")[-2],
        "LaiUhlrich2022_scaled_no_patella.osim",
    )
    generateVisualizerJson(
        modelPath=model_mono_file,
        ikPath=pathOutputMotion,
        jsonOutputPath=output_mono_json_path,
        vertical_offset=0,
    )

    # assert output_mono_json_path is created
    assert os.path.exists(output_mono_json_path), "Mono json file not created"

    if marker_wham_path is not None:
        output_wham_json_path = os.path.join(
            "/", *pathOutputMotion.split("/")[:-1], "wham.json"
        )
        model_wham_file = model_mono_file
        wham_file = pathOutputMotion.replace(".mot", "_wham.mot")
        assert (
            wham_file is not None
        ), "No wham_result.mot file found in the wham_result folder"

        generateVisualizerJson(
            modelPath=model_wham_file,
            ikPath=wham_file,
            jsonOutputPath=output_wham_json_path,
            vertical_offset=0,
        )

    # 2cams
    output_2cams_json_path = os.path.join(
        "/", *pathOutputMotion.split("/")[:-1], "2cams.json"
    )
    file_original_name = file
    if "_sync.mot" in file:
        file_original_name = file_original_name.split("_sync.mot")[0] + ".mot"

    twocams_file = os.path.join(
        f"LabValidation_withVideos1/{subject}/OpenSimData/Video/HRNet/2-cameras/IK/{file_original_name}"
    )
    model_2cams_file = os.path.join(
        f"LabValidation_withVideos1/{subject}/OpenSimData/Video/HRNet/2-cameras/Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim"
    )
    generateVisualizerJson(
        modelPath=model_2cams_file,
        ikPath=twocams_file,
        jsonOutputPath=output_2cams_json_path,
        vertical_offset=0,
    )

    logger.info("Json files created")

    output_video_path_mono = os.path.join(movement_path, "viewer_mono.webm")

    json_files = [output_mocap_json_path, output_mono_json_path]

    if marker_wham_path is not None:
        json_files.append(output_wham_json_path)

    if marker_wham_path is not None:
        json_files.append(output_2cams_json_path)

    viz = False
    if rerun and viz:
        automate_recording(json_files, output_video_path=output_video_path_mono)

    logger.info("Visualization(s) created")

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
    logger.info("Results path: " + results_path)

    # monitor.stop_monitoring()

    # # Create monitoring output directory
    # monitoring_dir = os.path.join(case_path, "monitoring")
    # os.makedirs(monitoring_dir, exist_ok=True)

    # # Save monitoring data and generate plots
    # data_path = os.path.join(monitoring_dir, "resource_usage_data.json")
    # monitor.save_data(data_path)

    # plot_path = os.path.join(monitoring_dir, "resource_usage_plot.png")
    # monitor.plot_usage(plot_path)

    # # Print summary
    # monitor.print_summary()

    # logger.info(f"Resource monitoring results saved to {monitoring_dir}")

    # # Create SMPL model from corrected markers and motion
    # logger.info("Creating SMPL model from corrected markers and motion")
    # smpl_output_path = os.path.join(case_path, "SMPL")
    # os.makedirs(smpl_output_path, exist_ok=True)

    # # Find the corrected TRC file
    # corrected_trc_path_folder = os.path.join(case_path, "MarkerData")
    # # find the folder which does not contain 'error' in the name
    # for folder in os.listdir(corrected_trc_path_folder):
    #     if "error" not in folder:
    #         corrected_trc_path_sub_folder = os.path.join(corrected_trc_path_folder, folder)
    #         break

    # # find the trc file containing '_sync.trc' in the name
    # for file in os.listdir(corrected_trc_path_sub_folder):
    #     if "_sync.trc" in file:
    #         corrected_trc_path = os.path.join(corrected_trc_path_sub_folder, file)
    #         break

    # # Find the corrected MOT file
    # corrected_mot_path = pathOutputMotion  # This is the IK result motion file

    # # Create SMPL model from markers and motion
    # marker_to_smpl = MarkerToSMPL(
    #     trc_path=corrected_trc_path,
    #     mot_path=corrected_mot_path,
    #     output_dir=smpl_output_path,
    #     gender="male" if sex == "m" else "female"
    # )

    # # Estimate SMPL parameters
    # smpl_params = marker_to_smpl.estimate_smpl_parameters()

    # # Save SMPL parameters
    # smpl_params_path = marker_to_smpl.save_smpl_params(smpl_params)
    # logger.info(f"SMPL parameters saved to {smpl_params_path}")

    # # Find the optimized results file
    # optimized_results_path = os.path.join(case_path, f"{video}_optimized.pkl")

    # if os.path.exists(optimized_results_path):
    #     # Generate visualization from optimized results (this works properly)
    #     viz_path_opt = marker_to_smpl.visualize_optimized_results(optimized_results_path)
    # else:
    #     # We'll skip the marker-based visualization since it has issues
    #     logger.warning("Optimized results not found. Skipping visualization.")
    #     # Alternatively, use the simplified matplotlib visualization
    #     viz_path = marker_to_smpl._generate_simple_visualization(smpl_params)
    #     logger.info(f"Simple SMPL visualization saved to {viz_path}")


if __name__ == "__main__":
    # Run the test flow
    flow()
