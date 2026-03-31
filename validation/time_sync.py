import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utilsChecker import cross_corr_multiple_timeseries


def read_mot_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith("time"):
            start_line = i
            break

    data = pd.read_csv(file_path, delimiter="\t", skiprows=start_line)
    return data


def pad_signals_half(Y1, Y2, pad_with="first_last"):
    max_len = max(Y1.shape[1], Y2.shape[1])
    # print('Y1 shape:', Y1.shape)
    # print('Y2 shape:', Y2.shape)

    if pad_with == "avg":
        pad_value_Y1_start = np.mean(Y1, axis=1)[:, None]
        pad_value_Y2_start = np.mean(Y2, axis=1)[:, None]
        pad_value_Y1_end = pad_value_Y1_start
        pad_value_Y2_end = pad_value_Y2_start
    elif pad_with == "first_last":
        pad_value_Y1_start = Y1[:, 0][:, None]
        pad_value_Y2_start = Y2[:, 0][:, None]
        pad_value_Y1_end = Y1[:, -1][:, None]
        pad_value_Y2_end = Y2[:, -1][:, None]
    else:
        raise ValueError("pad_with must be 'avg' or 'last'")

    if Y1.shape[1] < max_len:
        pad_width = max_len - Y1.shape[1]
        pad_width_start = pad_width // 2
        pad_width_end = pad_width - pad_width_start
        Y1 = np.hstack(
            [
                np.tile(pad_value_Y1_start, (1, pad_width_start)),
                Y1,
                np.tile(pad_value_Y1_end, (1, pad_width_end)),
            ]
        )
        # print(f"Padding Y1 with {pad_width_start} values at the start and {pad_width_end} values at the end using ")
        # print(pad_value_Y1_start, pad_value_Y1_end)
        # print('Y1 shape:', Y1.shape)
    if Y2.shape[1] < max_len:
        pad_width = max_len - Y2.shape[1]
        pad_width_start = pad_width // 2
        pad_width_end = pad_width - pad_width_start
        Y2 = np.hstack(
            [
                np.tile(pad_value_Y2_start, (1, pad_width_start)),
                Y2,
                np.tile(pad_value_Y2_end, (1, pad_width_end)),
            ]
        )
        # print(f"Padding Y2 with {pad_width_start} values at the start and {pad_width_end} values at the end using ")
        # print(pad_value_Y2_start, pad_value_Y2_end)
        # print('Y2 shape:', Y2.shape)

    return Y1, Y2


def pad_signals(Y1, Y2, pad_with="last"):
    max_len = max(Y1.shape[1], Y2.shape[1])
    # print('Y1 shape:', Y1.shape)
    # print('Y2 shape:', Y2.shape)

    if pad_with == "avg":
        pad_value_Y1 = np.mean(Y1, axis=1)[:, None]
        pad_value_Y2 = np.mean(Y2, axis=1)[:, None]
    elif pad_with == "last":
        pad_value_Y1 = Y1[:, -1][:, None]
        pad_value_Y2 = Y2[:, -1][:, None]
    else:
        raise ValueError("pad_with must be 'avg' or 'last'")

    if Y1.shape[1] < max_len:
        pad_width = max_len - Y1.shape[1]
        Y1 = np.hstack([Y1, np.tile(pad_value_Y1, (1, pad_width))])
        # print(f"Padding Y1 with {pad_width} values using ")
        # print(pad_value_Y1)
        # print('Y1 shape:', Y1.shape)
    if Y2.shape[1] < max_len:
        pad_width = max_len - Y2.shape[1]
        Y2 = np.hstack([Y2, np.tile(pad_value_Y2, (1, pad_width))])
        # print(f"Padding Y2 with {pad_width} values using ")
        # print(pad_value_Y2)
        # print('Y2 shape:', Y2.shape)

    return Y1, Y2


def get_array_of_angles(Y, angle_min=45, angle_max=46):
    return np.where((Y > angle_min) & (Y < angle_max))


def find_first_45_degrees(Y):
    knee_angle_r = Y[0]
    knee_angle_l = Y[1]

    knee_angle_r = np.abs(knee_angle_r)
    knee_angle_l = np.abs(knee_angle_l)

    # find the first index where the angle is approximately 45 degrees (use 45.5 +- 0.5)
    r_array = [[]]
    l_array = [[]]
    angle_min = 45
    angle_max = 45
    while len(r_array[0]) == 0:
        angle_min -= 1
        angle_max += 1
        r_array = get_array_of_angles(
            knee_angle_r, angle_min=angle_min, angle_max=angle_max
        )

    angle_min = 45
    angle_max = 45
    while len(l_array[0]) == 0:
        angle_min -= 1
        angle_max += 1
        l_array = get_array_of_angles(
            knee_angle_l, angle_min=angle_min, angle_max=angle_max
        )

    idx_r = r_array[0][0]
    idx_l = l_array[0][0]

    return idx_r, idx_l


def shift_time_series(Y1, Y2, lag):
    if lag > 0:
        shifted_Y2 = np.roll(Y2, lag, axis=1)
        shifted_Y2[:, :lag] = np.nan  # Set wrapped-around values to NaN
    elif lag < 0:
        shifted_Y2 = np.roll(Y2, lag, axis=1)
        shifted_Y2[:, lag:] = np.nan  # Set wrapped-around values to NaN
    else:
        shifted_Y2 = Y2

    shifted_Y1 = Y1  # Y1 remains unshifted
    return shifted_Y1, shifted_Y2


def run_time_sync(
    subject, session, camera, movement, output_case_path=None, visualize=False
):
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    movement_path = os.path.join(
        repo_path, "output", subject, session, camera, movement
    )
    if output_case_path is not None:
        movement_path = output_case_path

    fol_path_name = "LabValidation_withVideos1"
    validation_videos_path = os.path.join(repo_path, fol_path_name)
    output_path = os.path.join(repo_path, "output")
    graph_path = None

    OpenSimDataDir = os.path.join(validation_videos_path, subject, "OpenSimData")
    mocap_dir = os.path.join(OpenSimDataDir, "Mocap", "IK")
    subject_path = os.path.join(output_path, subject)

    mov_path = os.path.join(subject_path, session, camera, movement)
    if not os.path.isdir(mov_path):
        return None, None, None

    if output_case_path is None:
        mov_dir = [
            f for f in os.listdir(mov_path) if os.path.isdir(os.path.join(mov_path, f))
        ]

        if len(mov_dir) == 0:
            return None, None, None
        elif len(mov_dir) == 1:
            mov = mov_dir[0]
        elif any("trimmed" in folder for folder in mov_dir):
            mov = [folder for folder in mov_dir if "trimmed" in folder][0]
        else:
            return None, None, None

        movement_path = os.path.join(subject_path, session, camera, movement, mov)
        openSim_video_path = os.path.join(movement_path, "OpenSim")
        video_ik_path = os.path.join(openSim_video_path, "IK")

    else:
        movement_path = output_case_path

    if output_case_path is not None:
        video_ik_path = os.path.join(output_case_path, "OpenSim", "IK")

    if not os.path.exists(video_ik_path):
        return None, None, None

    video_dirs = os.listdir(video_ik_path)
    shifted_dir = os.path.join(video_ik_path, "shiftedIK")
    if not os.path.exists(shifted_dir):
        os.makedirs(shifted_dir)

    for file in os.listdir(shifted_dir):
        os.remove(os.path.join(shifted_dir, file))

    for video_dir in video_dirs:
        if "wham_result" in video_dir:
            continue
        video_dir_path = os.path.join(video_ik_path, video_dir)
        video_mot_files = [f for f in os.listdir(video_dir_path) if f.endswith(".mot")]
        modified_video_mot_files = [f.split("_")[0] + ".mot" for f in video_mot_files]

        for i, modified_video_file in enumerate(modified_video_mot_files):
            if "shifted" in modified_video_file:
                continue

            video_file = video_mot_files[i]
            mocap_file_path = os.path.join(mocap_dir, modified_video_file)
            video_file_path = os.path.join(video_dir_path, video_file)

            if "shifted" in video_file_path:
                continue

            # print(f'Mocap file: {mocap_file_path}')
            # print(f'Video file: {video_file_path}')

            mocap_data = read_mot_file(mocap_file_path)
            video_data = read_mot_file(video_file_path)

            mocap_df = pd.DataFrame(mocap_data)
            video_df = pd.DataFrame(video_data)

            video_df["time_"] = pd.to_timedelta(video_df["time"], unit="s")
            video_df = video_df.set_index("time_")
            target_freq = "10ms"
            video_df = video_df.resample(target_freq).interpolate(method="linear")
            video_df = video_df.reset_index()
            video_df["time"] = video_df["time_"].dt.total_seconds()
            video_df["time"] = video_df["time"].round(2)
            video_df = video_df.drop(columns=["time_"])

            mocap_df_untouched = mocap_df.copy()
            video_df_untouched_upsampled = video_df.copy()

            knee_ankle_columns = ["knee_angle_r", "knee_angle_l"]
            mocap_knee_ankle = mocap_df[knee_ankle_columns].values.T
            video_knee_ankle = video_df[knee_ankle_columns].values.T

            idx_r_mocap, idx_l_mocap = find_first_45_degrees(mocap_knee_ankle)
            idx_r_video, idx_l_video = find_first_45_degrees(video_knee_ankle)

            approximate_lag_right = idx_r_mocap - idx_r_video
            approximate_lag_left = idx_l_mocap - idx_l_video
            approximate_lag = (approximate_lag_right + approximate_lag_left) // 2

            # print("Y1 is mocap, Y2 is video")
            mocap_knee_ankle_padded, video_knee_ankle_padded = pad_signals(
                mocap_knee_ankle, video_knee_ankle, pad_with="avg"
            )

            max_corr, lag = cross_corr_multiple_timeseries(
                mocap_knee_ankle_padded,
                video_knee_ankle_padded,
                visualize=visualize,
                frameRate=100,
                multCorrGaussianStd=2000,
                path=shifted_dir,
                approximateLag=approximate_lag_right,
            )

            max_corr = round(max_corr, 3)

            # if session == 'Session1':
            #     lag += 2

            shifted_mocap, shifted_video = shift_time_series(
                mocap_knee_ankle, video_knee_ankle, lag
            )

            shifted_mocap_df = pd.DataFrame(shifted_mocap.T, columns=knee_ankle_columns)
            shifted_video_df = pd.DataFrame(shifted_video.T, columns=knee_ankle_columns)

            single_plot = True
            multi_plot = True

            modified_video_file = modified_video_file.split(".")[0]

            if single_plot:
                fig_s = go.Figure()
                fig_s.add_trace(
                    go.Scatter(
                        x=shifted_mocap_df.index,
                        y=shifted_mocap_df["knee_angle_r"],
                        mode="lines",
                        name="Mocap",
                        line=dict(color="#1f77b4", width=2),  # Blue for Mocap
                    )
                )
                fig_s.add_trace(
                    go.Scatter(
                        x=shifted_mocap_df.index,
                        y=shifted_video_df["knee_angle_r"],
                        mode="lines",
                        name="Video",
                        line=dict(color="#ff7f0e", width=2),  # Orange for Video
                    )
                )
                fig_s.update_layout(
                    title=f"Mocap vs Video Knee Angle Right - {movement} - {video_dir}. Lag: {lag} - Correlation: {max_corr}",
                    xaxis_title="Time",
                    yaxis_title="Angle (deg)",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    showlegend=True,
                )

            if multi_plot:
                # Define color schemes
                mocap_color = "#1f77b4"  # Blue
                video_color = "#ff7f0e"  # Orange

                fig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True, subplot_titles=knee_ankle_columns
                )

                for i, sensor in enumerate(knee_ankle_columns, start=1):
                    fig.add_trace(
                        go.Scatter(
                            x=shifted_mocap_df.index,
                            y=shifted_mocap_df[sensor],
                            mode="lines",
                            name=f"Mocap {sensor}",
                            line=dict(color=mocap_color, width=2),
                        ),
                        row=i,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=shifted_mocap_df.index,
                            y=shifted_video_df[sensor],
                            mode="lines",
                            name=f"Video {sensor}",
                            line=dict(color=video_color, width=2),
                        ),
                        row=i,
                        col=1,
                    )

                fig.update_layout(
                    title=f"Mocap vs Video Angles - {movement} - {video_dir}. Lag: {lag} - Correlation: {max_corr}",
                    xaxis_title="Index",
                    yaxis_title="Angle (deg)",
                    height=800,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    showlegend=True,
                )

                # Update all subplot axes for better visibility
                for i in range(1, 5):
                    fig.update_xaxes(
                        showgrid=True, gridwidth=1, gridcolor="LightGray", row=i, col=1
                    )
                    fig.update_yaxes(
                        showgrid=True, gridwidth=1, gridcolor="LightGray", row=i, col=1
                    )

                graph_path = os.path.join(
                    shifted_dir, f"lag_correlation_{modified_video_file}.html"
                )
                fig.write_html(graph_path)

                fig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True, subplot_titles=knee_ankle_columns
                )

                for i, sensor in enumerate(knee_ankle_columns, start=1):
                    fig.add_trace(
                        go.Scatter(
                            x=mocap_df_untouched.index,
                            y=mocap_df_untouched[sensor],
                            mode="lines",
                            name=f"Mocap {sensor}",
                        ),
                        row=i,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=video_df_untouched_upsampled.index,
                            y=video_df_untouched_upsampled[sensor],
                            mode="lines",
                            name=f"Video {sensor}",
                        ),
                        row=i,
                        col=1,
                    )

                fig.update_layout(
                    title=f"Mocap vs Video Angles (Unshifted) - {movement} - {video_dir}. Lag: {lag} - Correlation: {max_corr}",
                    xaxis_title="Index",
                    yaxis_title="Angle (deg)",
                    height=800,
                )

                fig.write_html(
                    os.path.join(
                        shifted_dir,
                        f"unshifted_lag_correlation_{modified_video_file}.html",
                    )
                )

                fig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True, subplot_titles=knee_ankle_columns
                )
                for i, sensor in enumerate(knee_ankle_columns, start=1):
                    fig.add_trace(
                        go.Scatter(
                            x=np.arange(mocap_knee_ankle_padded.shape[1]),
                            y=mocap_knee_ankle_padded[i - 1],
                            mode="lines",
                            name=f"Mocap {sensor}",
                        ),
                        row=i,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=np.arange(video_knee_ankle_padded.shape[1]),
                            y=video_knee_ankle_padded[i - 1],
                            mode="lines",
                            name=f"Video {sensor}",
                        ),
                        row=i,
                        col=1,
                    )
                fig.update_layout(
                    title=f"Mocap vs Video Angles (Padded) - {movement} - {video_dir}. Lag: {lag} - Correlation: {max_corr}",
                    xaxis_title="Index",
                    yaxis_title="Angle (deg)",
                    height=800,
                )
                fig.write_html(
                    os.path.join(shifted_dir, f"padded_{modified_video_file}.html")
                )

            lag_file = os.path.join(
                shifted_dir, f"lag_correlation_{modified_video_file}.txt"
            )

            with open((lag_file), "w") as f:
                f.write(
                    f"Lag: {lag}\nCorrelation: {max_corr}\nAproximate Lag: {approximate_lag}\nright_knee_45_mocap: {idx_r_mocap}\nleft_knee_45_mocap: {idx_l_mocap}\nright_knee_45_video: {idx_r_video}\nleft_knee_45_video: {idx_l_video}"
                )
            # print(f'Lag and correlation values written to {lag_file}')
            # print('--------------------------------------------')

            return lag, max_corr, graph_path


# Example usage
# run_single_file('subject1', 'Session1', 'Cam1', 'walking')
