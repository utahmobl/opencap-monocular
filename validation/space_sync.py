import numpy as np
import plotly.graph_objects as go
from utils.utils_trc import (
    write_trc,
    TRCFile,
    transform_from_tuple_array,
    align_trc_files,
)
import os
from loguru import logger
from utils.opensim.utils_opensim import runIKTool


def calculate_midpoint(trc_file, right_marker, left_marker):
    assert trc_file.marker_exists(
        right_marker
    ), f"Marker {right_marker} does not exist in the TRC file."
    assert trc_file.marker_exists(
        left_marker
    ), f"Marker {left_marker} does not exist in the TRC file."

    right_data = trc_file.marker(right_marker)
    left_data = trc_file.marker(left_marker)

    midpoint = np.array(
        [
            (right_data[:, 0] + left_data[:, 0]) / 2,
            (right_data[:, 1] + left_data[:, 1]) / 2,
            (right_data[:, 2] + left_data[:, 2]) / 2,
        ]
    ).T

    return midpoint


def run_space_sync_and_ik(
    subject, session, camera, movement, marker_wham_path=False, output_case_path=None
):

    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    validation_videos_path = os.path.join(repo_path, "LabValidation_withVideos1")
    output_path = os.path.join(repo_path, "output")

    frame_rate = 100

    MarkerDataDir = os.path.join(validation_videos_path, subject, "MarkerData", "Mocap")
    subject_path = os.path.join(output_path, subject)

    if output_case_path is None:
        movement_folder = os.path.join(subject_path, session, camera, movement)
        # if output_case_path is not None:
        #     movement_folder = output_case_path

        movement_folders = [
            f
            for f in os.listdir(movement_folder)
            if os.path.isdir(os.path.join(movement_folder, f))
        ]

        if not movement_folders:
            logger.info(f"Skipping {movement_folder} as it is empty.")
            return None, None, None

        if len(movement_folders) > 1:
            trimmed = next((d for d in movement_folders if "trimmed" in d), None)
            if trimmed is None:
                logger.info(f"Skipping {movement_folder} as it is empty.")
                return None, None, None
            movement_path = os.path.join(movement_folder, trimmed)
        else:
            movement_path = os.path.join(movement_folder, movement_folders[0])

        marker_video_path = os.path.join(movement_path, "MarkerData")
        error_markers_path = os.path.join(movement_path, "MarkerData")
    else:
        movement_path = output_case_path
        marker_video_path = os.path.join(output_case_path, "MarkerData")
        error_markers_path = os.path.join(output_case_path, "MarkerData")

    if not os.path.exists(marker_video_path):
        return None, None, None

    marker_video_subdirs = [
        d for d in os.listdir(marker_video_path) if d != "errors" and d != "shiftedIK"
    ]
    if not marker_video_subdirs:
        return None, None, None

    logger.info(f"Filepath: {marker_video_path}")

    folder = next(
        (d for d in marker_video_subdirs if "." not in d and "wham_result" not in d),
        None,
    )
    if folder is None:
        return None, None, None

    marker_video_path = os.path.join(marker_video_path, folder)
    marker_video_files = [
        f for f in os.listdir(marker_video_path) if f.endswith(".trc")
    ]
    if not marker_video_files:
        return None, None, None

    for file in marker_video_files:
        if "_sync.trc" in file:
            os.remove(os.path.join(marker_video_path, file))
            marker_video_files.remove(file)

    marker_video_path = os.path.join(marker_video_path, marker_video_files[0])

    movement_file_name_trc = movement + ".trc"
    marker_mocap_path = os.path.join(MarkerDataDir, movement_file_name_trc)

    trc_mono = TRCFile(marker_video_path)
    trc_mocap = TRCFile(marker_mocap_path)
    if marker_wham_path:
        trc_wham = TRCFile(marker_wham_path)

    mocap_start, mocap_end = trc_mocap.get_start_end_times()
    mono_start, mono_end = trc_mono.get_start_end_times()

    logger.info(f"Mocap Start: {mocap_start}, Mocap End: {mocap_end}")
    logger.info(f"Mono Start: {mono_start}, Mono End: {mono_end}")

    lag_file_name = f"lag_correlation_{movement}.txt"
    lag_file_path = os.path.join(
        movement_path, "OpenSim", "IK", "shiftedIK", lag_file_name
    )
    if not os.path.exists(lag_file_path):
        return None, None, None

    with open(lag_file_path, "r") as file:
        lag = int(file.readline().split(":")[-1])
        correlation = float(file.readline().split(":")[-1])

    if trc_mono.get_frequency() != frame_rate:
        logger.info(f"Resampling mono TRC to {frame_rate} Hz.")
        trc_mono.resample_trc(target_frequency=frame_rate)

    if trc_mocap.get_frequency() != frame_rate:
        logger.info(f"Resampling mocap TRC to {frame_rate} Hz.")
        trc_mocap.resample_trc(target_frequency=frame_rate)

    if marker_wham_path:
        if trc_wham.get_frequency() != frame_rate:
            logger.info(f"Resampling wham TRC to {frame_rate} Hz.")
            trc_wham.resample_trc(target_frequency=frame_rate)

    align_trc_files(trc_mono, trc_mocap, lag)
    logger.info(f"Shifted mono TRC by {lag} frames.")

    if marker_wham_path:
        align_trc_files(trc_wham, trc_mocap, lag)
        logger.info(f"Shifted wham TRC by {lag} frames.")

    mocap_start, mocap_end = trc_mocap.get_start_end_times()
    mono_start, mono_end = trc_mono.get_start_end_times()
    if marker_wham_path:
        wham_start, wham_end = trc_wham.get_start_end_times()

    logger.info(f"Mocap Start: {mocap_start}, Mocap End: {mocap_end}")
    logger.info(f"Mono Start: {mono_start}, Mono End: {mono_end}")
    if marker_wham_path:
        logger.info(f"Wham Start: {wham_start}, Wham End: {wham_end}")

    trc_mono_marker_names = trc_mono.get_marker_names()
    trc_mocap_marker_names = trc_mocap.get_marker_names()

    mono_metric = trc_mono.get_metric_trc()
    mocap_metric = trc_mocap.get_metric_trc()
    if marker_wham_path:
        wham_metric = trc_wham.get_metric_trc()

    if mono_metric != "mm":
        trc_mono.convert_to_metric_trc(current_metric=mono_metric, target_metric="mm")
        logger.info(f"Converted mono markers from {mono_metric} to mm.")
    if mocap_metric != "mm":
        trc_mocap.convert_to_metric_trc(current_metric=mocap_metric, target_metric="mm")
        logger.info(f"Converted mocap markers from {mocap_metric} to mm.")
    if marker_wham_path:
        if wham_metric != "mm":
            trc_wham.convert_to_metric_trc(
                current_metric=wham_metric, target_metric="mm"
            )
            logger.info(f"Converted wham markers from {wham_metric} to mm.")

    trc_mocap_trimmed = trc_mocap.copy()
    trc_mocap_trimmed.trim_to_match(mono_start, mono_end)

    if marker_wham_path:
        trc_wham_trimmed = trc_wham.copy()
        trc_wham_trimmed.trim_to_match(mono_start, mono_end)
        trc_wham = trc_wham_trimmed

    mid_PSIS_mono = calculate_midpoint(trc_mono, "r_PSIS", "l_PSIS")
    mid_ASIS_mono = calculate_midpoint(trc_mono, "r_ASIS", "l_ASIS")

    mid_PSIS_mocap = calculate_midpoint(trc_mocap_trimmed, "r.PSIS", "L.PSIS")
    mid_ASIS_mocap = calculate_midpoint(trc_mocap_trimmed, "r.ASIS", "L.ASIS")
    if marker_wham_path:
        mid_PSIS_wham = calculate_midpoint(trc_wham, "r_PSIS", "l_PSIS")
        mid_ASIS_wham = calculate_midpoint(trc_wham, "r_ASIS", "l_ASIS")

    heading_vec_mono = mid_ASIS_mono - mid_PSIS_mono
    heading_vec_mocap = mid_ASIS_mocap - mid_PSIS_mocap
    if marker_wham_path:
        heading_vec_wham = mid_ASIS_wham - mid_PSIS_wham

    heading_vec_mono[:, 1] = 0
    heading_vec_mocap[:, 1] = 0
    if marker_wham_path:
        heading_vec_wham[:, 1] = 0

    heading_vec_mono_normalized = heading_vec_mono / np.linalg.norm(
        heading_vec_mono, axis=1, keepdims=True
    )
    heading_vec_mocap_normalized = heading_vec_mocap / np.linalg.norm(
        heading_vec_mocap, axis=1, keepdims=True
    )
    if marker_wham_path:
        heading_vec_wham_normalized = heading_vec_wham / np.linalg.norm(
            heading_vec_wham, axis=1, keepdims=True
        )

    dot_products = np.einsum(
        "ij,ij->i", heading_vec_mono_normalized, heading_vec_mocap_normalized
    )
    if marker_wham_path:
        dot_products_wham = np.einsum(
            "ij,ij->i", heading_vec_mono_normalized, heading_vec_wham_normalized
        )

    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
    if marker_wham_path:
        angles_wham = np.arccos(np.clip(dot_products_wham, -1.0, 1.0))

    angles_degrees = np.degrees(angles)
    if marker_wham_path:
        angles_degrees_wham = np.degrees(angles_wham)

    average_difference = np.mean(angles_degrees)
    if marker_wham_path:
        average_difference_wham = np.mean(angles_degrees_wham)

    logger.info(f"Average Angular Difference: {average_difference:.2f} degrees")
    if marker_wham_path:
        logger.info(
            f"Average Angular Difference Wham: {average_difference_wham:.2f} degrees"
        )

    trc_mono.rotate(axis="y", value=average_difference)
    if marker_wham_path:
        trc_wham.rotate(axis="y", value=average_difference)

    markers = {
        "r_knee": "r_knee",
        "l_knee": "L_knee",
        "r_ankle": "r_ankle",
        "l_ankle": "L_ankle",
        "r_shoulder": "R_Shoulder",
        "l_shoulder": "L_Shoulder",
        "r_ASIS": "r.ASIS",
        "l_ASIS": "L.ASIS",
        "r_PSIS": "r.PSIS",
        "l_PSIS": "L.PSIS",
    }

    markers_wham = {
        "r_knee": "r_knee",
        "l_knee": "L_knee",
        "r_ankle": "r_ankle",
        "l_ankle": "L_ankle",
        "r_shoulder": "R_Shoulder",
        "l_shoulder": "L_Shoulder",
        "r_ASIS": "r.ASIS",
        "l_ASIS": "L.ASIS",
        "r_PSIS": "r.PSIS",
        "l_PSIS": "L.PSIS",
    }

    offsets_x, offsets_y, offsets_z = [], [], []
    if marker_wham_path:
        offsets_x_wham, offsets_y_wham, offsets_z_wham = [], [], []

    idx = 0

    for mono_marker, mocap_marker in markers.items():
        assert trc_mono.marker_exists(
            mono_marker
        ), f"Marker {mono_marker} does not exist in the mono TRC file."
        assert trc_mocap_trimmed.marker_exists(
            mocap_marker
        ), f"Marker {mocap_marker} does not exist in the mocap TRC file."

        mono_data = trc_mono.marker(mono_marker)
        mocap_data = trc_mocap_trimmed.marker(mocap_marker)

        offsets_x.append(mocap_data[idx, 0] - mono_data[0, 0])
        offsets_y.append(mocap_data[idx, 1] - mono_data[0, 1])
        offsets_z.append(mocap_data[idx, 2] - mono_data[0, 2])

    if marker_wham_path:
        for wham_marker, mocap_marker in markers_wham.items():
            assert trc_wham.marker_exists(
                wham_marker
            ), f"Marker {wham_marker} does not exist in the wham TRC file."
            assert trc_mocap_trimmed.marker_exists(
                mocap_marker
            ), f"Marker {mocap_marker} does not exist in the mocap TRC file."

            wham_data = trc_wham.marker(wham_marker)
            mocap_data = trc_mocap_trimmed.marker(mocap_marker)

            offsets_x_wham.append(mocap_data[idx, 0] - wham_data[0, 0])
            offsets_y_wham.append(mocap_data[idx, 1] - wham_data[0, 1])
            offsets_z_wham.append(mocap_data[idx, 2] - wham_data[0, 2])

    avg_x_offset = np.mean(offsets_x)
    avg_y_offset = np.mean(offsets_y)
    avg_z_offset = np.mean(offsets_z)

    if marker_wham_path:
        avg_x_offset_wham = np.mean(offsets_x_wham)
        avg_y_offset_wham = np.mean(offsets_y_wham)
        avg_z_offset_wham = np.mean(offsets_z_wham)

    logger.info(f"Average X Offset: {avg_x_offset:.2f} mm")
    logger.info(f"Average Y Offset: {avg_y_offset:.2f} mm")
    logger.info(f"Average Z Offset: {avg_z_offset:.2f} mm")

    trc_mono.offset(axis="x", value=avg_x_offset)
    trc_mono.offset(axis="y", value=avg_y_offset)
    trc_mono.offset(axis="z", value=avg_z_offset)

    if marker_wham_path:
        trc_wham.offset(axis="x", value=avg_x_offset_wham)
        trc_wham.offset(axis="y", value=avg_y_offset_wham)
        trc_wham.offset(axis="z", value=avg_z_offset_wham)

    marker_errors = {}
    for mono_marker, mocap_marker in markers.items():
        assert trc_mono.marker_exists(
            mono_marker
        ), f"Marker {mono_marker} does not exist in the mono TRC file."
        assert trc_mocap_trimmed.marker_exists(
            mocap_marker
        ), f"Marker {mocap_marker} does not exist in the mocap TRC file."

        mono_data = trc_mono.marker(mono_marker)
        mocap_data = trc_mocap_trimmed.marker(mocap_marker)

        errors = np.linalg.norm(mono_data - mocap_data, axis=1)
        marker_errors[mono_marker] = np.mean(errors)

    average_error = np.mean(list(marker_errors.values()))

    # logger.info(f"Marker Errors: {marker_errors}")
    # logger.info(f"Average Error: {average_error:.2f}")

    error_file_name = f"marker_errors_{movement}.csv"
    error_markers_path_file = os.path.join(
        error_markers_path, "errors", error_file_name
    )

    if marker_wham_path:
        marker_errors_wham = {}
        for wham_marker, mocap_marker in markers.items():
            assert trc_wham.marker_exists(
                wham_marker
            ), f"Marker {wham_marker} does not exist in the wham TRC file."
            assert trc_mocap_trimmed.marker_exists(
                mocap_marker
            ), f"Marker {mocap_marker} does not exist in the mocap TRC file."

            wham_data = trc_wham.marker(wham_marker)
            mocap_data = trc_mocap_trimmed.marker(mocap_marker)
            errors_wham = np.linalg.norm(wham_data - mocap_data, axis=1)
            marker_errors_wham[wham_marker] = np.mean(errors_wham)

        average_error_wham = np.mean(list(marker_errors_wham.values()))
        logger.info(f"Average Error Wham: {average_error_wham:.2f}")

    if not os.path.exists(os.path.join(error_markers_path, "errors")):
        os.makedirs(os.path.join(error_markers_path, "errors"))

    with open(error_markers_path_file, "w") as file:
        file.write("Marker,Error\n")
        for marker, error in marker_errors.items():
            file.write(f"{marker},{error}\n")
        file.write(f"Average,{average_error}\n")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(marker_errors.keys()),
            y=list(marker_errors.values()),
            name="Marker Errors (mm)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(marker_errors.keys()),
            y=[average_error] * len(marker_errors),
            mode="lines",
            name="Average Error (mm)",
        )
    )
    fig.update_layout(
        title=f"Marker Errors for {movement} (mm)",
        xaxis_title="Marker",
        yaxis_title="Error (mm)",
    )

    error_plot_file_name = f"marker_errors_plot_{movement}.html"
    error_plot_file_path = os.path.join(
        error_markers_path, "errors", error_plot_file_name
    )
    fig.write_html(error_plot_file_path)

    synced_path = marker_video_path.replace(".trc", "_sync.trc")

    if marker_wham_path:
        synced_path_wham = marker_wham_path.replace("wham_result.trc", "sync_wham.trc")

    start_time, end_time = trc_mono.get_start_end_times()
    if marker_wham_path:
        start_time_wham, end_time_wham = trc_wham.get_start_end_times()

    logger.info(f"Synced Mono Start Time: {start_time}")
    logger.info(f"Synced Mono End Time: {end_time}")
    if marker_wham_path:
        logger.info(f"Synced Wham Start Time: {start_time_wham}")
        logger.info(f"Synced Wham End Time: {end_time_wham}")

    mono_synced = transform_from_tuple_array(trc_mono.data)
    if marker_wham_path:
        wham_synced = transform_from_tuple_array(trc_wham.data)

    write_trc(
        keypoints3D=mono_synced,
        pathOutputFile=synced_path,
        keypointNames=trc_mono_marker_names,
        frameRate=frame_rate,
        unit="mm",
        t_start=start_time,
    )

    if marker_wham_path:
        write_trc(
            keypoints3D=wham_synced,
            pathOutputFile=synced_path_wham,
            keypointNames=trc_mono_marker_names,
            frameRate=frame_rate,
            unit="mm",
            t_start=start_time_wham,
        )

    trc_synced = TRCFile(synced_path)
    trc_synced_marker_names = trc_synced.get_marker_names()

    if marker_wham_path:
        trc_synced_wham = TRCFile(synced_path_wham)
        trc_synced_wham_marker_names = trc_synced_wham.get_marker_names()

    logger.info(f"Wrote synced marker data to: {synced_path}")
    if marker_wham_path:
        logger.info(f"Wrote synced wham marker data to: {synced_path_wham}")

    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pathGenericSetupFile = os.path.join(
        repo_path, "utils", "opensim", "IK", "Setup_IK_SMPL.xml"
    )

    pathOutputMotion = runIKTool(
        pathGenericSetupFile=pathGenericSetupFile,
        pathScaledModel=os.path.join(
            movement_path,
            "OpenSim",
            "Model",
            folder,
            "LaiUhlrich2022_scaled_no_patella.osim",
        ),
        pathTRCFile=synced_path,
        pathOutputFolder=os.path.join(movement_path, "OpenSim", "IK", "shiftedIK"),
    )

    logger.info(f"Ran IK on synced data. Results saved to: {pathOutputMotion}")

    if marker_wham_path:
        pathOutputMotion_wham = runIKTool(
            pathGenericSetupFile=pathGenericSetupFile,
            pathScaledModel=os.path.join(
                movement_path,
                "OpenSim",
                "Model",
                folder,
                "LaiUhlrich2022_scaled_no_patella.osim",
            ),
            pathTRCFile=synced_path_wham,
            pathOutputFolder=os.path.join(movement_path, "OpenSim", "IK", "shiftedIK"),
        )

        logger.info(
            f"Ran IK on synced wham data. Results saved to: {pathOutputMotion_wham}"
        )

    mocap_folder = os.path.join(movement_path, "mocap")
    if not os.path.exists(mocap_folder):
        os.makedirs(mocap_folder)

    os.system(f"cp {marker_mocap_path} {mocap_folder}")

    mocap_model_path = marker_mocap_path.replace("MarkerData", "OpenSimData")
    mocap_model_path = os.path.dirname(mocap_model_path)
    mocap_ik_path = mocap_model_path
    mocap_model_path = os.path.join(mocap_model_path, "Model")
    mocap_model_file = next(
        (f for f in os.listdir(mocap_model_path) if "scaled.osim" in f), None
    )
    mocap_model_path = os.path.join(mocap_model_path, mocap_model_file)
    os.system(f"cp {mocap_model_path} {mocap_folder}")
    mocap_ik_path = os.path.join(mocap_ik_path, "IK")
    ik_name = movement + ".mot"
    mocap_ik_path = os.path.join(mocap_ik_path, ik_name)
    os.system(f"cp {mocap_ik_path} {mocap_folder}")
    logger.info(f"Copied mocap data to: {mocap_folder}")
    logger.info("-" * 70)

    # TODO return the results and path
    return movement_path, synced_path, pathOutputMotion


if __name__ == "__main__":
    run_space_sync_and_ik("subject3", "Session1", "Cam1", "walking1", wham=False)
