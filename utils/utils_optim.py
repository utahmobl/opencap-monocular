import sys
import os
import pickle
import subprocess  # Added for ffmpeg
import warnings

sys.path.append("../")
import joblib
import torch
from scipy.signal import butter, filtfilt
import cv2
from loguru import logger
from typing import Optional

# from slahmr
import slahmr.slahmr.geometry.camera as camera
from slahmr.slahmr.body_model import (
    SMPL_JOINTS,
    KEYPT_VERTS,
    smpl_to_openpose,
    run_smpl,
)
from slahmr.slahmr.util.loaders import load_smpl_body_model


# SMPL path
# repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# smpl_model_path = os.path.join(
#     repo_path, "WHAM", "dataset", "body_models", "smpl", "SMPL_NEUTRAL.pkl"
# )  # use SMPLX b/c we have the mapping to openpose joints


def save_keypoints(data_dir, keypoints):
    keypoints_file = os.path.join(data_dir, "key2d.pth")
    torch.save(keypoints, keypoints_file)


def get_video_info(data_dir=None, video_path=None, release=False):
    if video_path:
        video_file = video_path
    else:
        video_files = [
            f
            for f in os.listdir(data_dir)
            if (f.endswith((".mp4", ".avi", ".mov")))
            and "sync" not in f  # Added .mov and ensure tuple for endswith
        ]
        if not video_files:
            raise FileNotFoundError(
                "No suitable video files found in the directory {}".format(data_dir)
            )
        video_file = os.path.join(data_dir, video_files[0])

    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if release:
        cap.release()
        return None, frame_rate, n_frames, video_file

    return cap, frame_rate, n_frames, video_file


# utilities
def load_results(result_path):
    results = joblib.load(result_path)

    return results


def get_openpose_keypoints(
    keypoints_wholebody, filter_freq=None, sample_rate=60, device="cpu"
):
    # keypoints_wholebody is a numpy array of shape (N, 133, 3)

    # openpose_keypoints is a numpy array of shape (N, 25, 3)
    openpose_keypoints = torch.empty(
        (keypoints_wholebody.shape[0], 25, 3), device=device, dtype=torch.float32
    )

    # 25 keypoints
    openpose_keypoints[
        :,
        [
            0,
            16,
            15,
            18,
            17,
            5,
            2,
            6,
            3,
            7,
            4,
            12,
            9,
            13,
            10,
            14,
            11,
            19,
            20,
            21,
            22,
            23,
            24,
        ],
        :,
    ] = torch.from_numpy(keypoints_wholebody[:, :23, :]).to(
        device=device, dtype=torch.float32
    )

    # Add mid-hip and Neck
    openpose_keypoints[:, 1, :] = torch.mean(openpose_keypoints[:, [2, 5], :], axis=1)
    openpose_keypoints[:, 8, :] = torch.mean(openpose_keypoints[:, [9, 12], :], axis=1)

    if filter_freq is not None:
        output_filter = filter_2d_keypoints(
            openpose_keypoints.cpu().numpy()[:, :, :2], 4, filter_freq, sample_rate
        )
        openpose_keypoints[:, :, :2] = torch.from_numpy(output_filter).to(device)

    return openpose_keypoints


def filter_2d_keypoints(trajectory, order, cutoff_freq, sampling_rate):
    """
    Apply a Butterworth filter to a 2D point trajectory.

    :param trajectory: numpy array of shape (time,nPoints, 2) representing the 2D points
    :param order: Order of the Butterworth filter
    :param cutoff_freq: Cutoff frequency of the filter in Hz
    :param sampling_rate: Sampling rate of the trajectory in Hz
    :return: Filtered trajectory
    """
    # Create a Butterworth filter. filt filt doubles the order
    b, a = butter(N=order / 2, Wn=cutoff_freq / (0.5 * sampling_rate), btype="low")

    # Apply the filter to each dimension independently
    filtered_trajectory = np.empty_like(trajectory)
    for i in range(trajectory.shape[1]):
        filtered_trajectory[:, i, 0] = filtfilt(
            b, a, trajectory[:, i, 0]
        )  # Filter x-coordinates
        filtered_trajectory[:, i, 1] = filtfilt(
            b, a, trajectory[:, i, 1]
        )  # Filter y-coordinates

    return filtered_trajectory


def filter_array(trajectory, order, cutoff_freq, sampling_rate):
    """
    Apply a Butterworth filter to the columns of a numpy array.

    :param trajectory: numpy array of shape (time,nPoints)
    :param order: Order of the Butterworth filter
    :param cutoff_freq: Cutoff frequency of the filter in Hz
    :param sampling_rate: Sampling rate of the trajectory in Hz
    :return: Filtered trajectory
    """
    # Create a Butterworth filter. filt filt doubles the order
    b, a = butter(N=order / 2, Wn=cutoff_freq / (0.5 * sampling_rate), btype="low")

    # Apply the filter to each dimension independently
    filtered_trajectory = np.empty_like(trajectory)
    for i in range(trajectory.shape[1]):
        filtered_trajectory[:, i] = filtfilt(
            b, a, trajectory[:, i]
        )  # Filter x-coordinates
        filtered_trajectory[:, i] = filtfilt(
            b, a, trajectory[:, i]
        )  # Filter y-coordinates

    return filtered_trajectory


def load_intrinsics(calib_path):
    if calib_path.endswith(".pickle"):  # opencap calibration
        with open(calib_path, "rb") as file:
            intrinsics_dict = pickle.load(file)
        intrinsics = {
            "fx": intrinsics_dict["intrinsicMat"][0, 0],
            "fy": intrinsics_dict["intrinsicMat"][1, 1],
            "cx": intrinsics_dict["intrinsicMat"][0, 2],
            "cy": intrinsics_dict["intrinsicMat"][1, 2],
            "k1": intrinsics_dict["distortion"][0, 0],
            "k2": intrinsics_dict["distortion"][0, 1],
            "k3": intrinsics_dict["distortion"][0, 4],
            "p1": intrinsics_dict["distortion"][0, 2],
            "p2": intrinsics_dict["distortion"][0, 3],
        }
        # Store portrait calibration dimensions for resolution-scaling in optimization.
        # imageSize is [[height], [width]] (portrait orientation) in these pickles.
        if "imageSize" in intrinsics_dict:
            calib_sz = np.array(intrinsics_dict["imageSize"]).flatten()
            intrinsics["calib_portrait_h"] = float(calib_sz[0])
            intrinsics["calib_portrait_w"] = float(calib_sz[1])

    else:
        calib = np.loadtxt(calib_path, delimiter=" ")
        fx, fy, cx, cy = calib[:4]
        intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

        if calib.shape[0] > 4:
            k1, k2, k3 = calib[4:]
            intrinsics["k1"] = k1
            intrinsics["k2"] = k2
            intrinsics["k3"] = k3

    return intrinsics


def to_torch(obj, device="cpu"):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float().to(device)
    if isinstance(obj, dict):
        return {k: to_torch(v).to(device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_torch(x).to(device) for x in obj]
    return obj


def pred_smpl(body_model, trans, root_orient, body_pose, betas):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : B x D
    """
    smpl2op_map = smpl_to_openpose(
        "smpl",
        use_hands=False,
        use_face=False,
        use_face_contour=False,
        openpose_format="coco25",
    )

    smpl_out = run_smpl(body_model, trans, root_orient, body_pose, betas)
    joints3d, points3d = smpl_out["joints"], smpl_out["vertices"]

    # select desired joints and vertices
    joints3d_body = joints3d[:, :, : len(SMPL_JOINTS), :]
    extra_vertices = joints3d[:, :, len(SMPL_JOINTS) :, :]
    joints3d_op = joints3d[:, :, smpl2op_map, :]
    # hacky way to get hip joints that align with ViTPose keypoints
    # this could be moved elsewhere in the future (and done properly)
    joints3d_op[:, :, [9, 12]] = (
        joints3d_op[:, :, [9, 12]]
        + 0.25 * (joints3d_op[:, :, [9, 12]] - joints3d_op[:, :, [12, 9]])
        + 0.5
        * (
            joints3d_op[:, :, [8]]
            - 0.5 * (joints3d_op[:, :, [9, 12]] + joints3d_op[:, :, [12, 9]])
        )
    )

    # Create mid hip and neck markers
    joints3d_op[:, :, 1, :] = torch.mean(joints3d_op[:, :, [2, 5]], axis=2)
    joints3d_op[:, :, 8, :] = torch.mean(joints3d_op[:, :, [9, 12]], axis=2)

    verts3d = points3d[:, :, KEYPT_VERTS, :]

    return {
        "points3d": points3d,  # all vertices
        "verts3d": verts3d,  # keypoint vertices
        "joints3d": joints3d_body,  # smpl joints
        "extra_vertices": extra_vertices,  # extra vertices that we defined
        "joints3d_op": joints3d_op,  # OP joints
        "faces": smpl_out["faces"],  # index array of faces
        "verts3d_all": points3d,
    }


def compute_mean_beta(betas, percentage_buffer=0.5):
    # find the mean beta parameters of the middle percentage of betas

    # Assuming betas is a PyTorch tensor
    average_beta = torch.mean(betas, dim=0)

    n_betas_middle = int(round(percentage_buffer * betas.size(0)))
    betas_middle = torch.zeros((n_betas_middle, betas.size(1)), device=betas.device)

    for i in range(n_betas_middle):
        diff = betas - average_beta
        dist = torch.norm(diff, dim=1)
        closest_beta_idx = torch.argmin(dist)
        betas_middle[i] = betas[closest_beta_idx]
        betas = torch.cat((betas[:closest_beta_idx], betas[closest_beta_idx + 1 :]))

    average_beta_percent = torch.mean(betas_middle, dim=0)

    return average_beta_percent


def load_smpl(model_type="smpl", device="cpu", gender="neutral"):
    # Construct the path based on gender
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    smpl_dir = os.path.join(repo_path, "WHAM", "dataset", "body_models", "smpl")

    # if gender == "male":
    #     model_filename = "SMPL_MALE.pkl"
    #     logger.info(f"Loading SMPL model for male: {model_filename}")
    # elif gender == "female":
    #     model_filename = "SMPL_FEMALE.pkl"
    #     logger.info(f"Loading SMPL model for female: {model_filename}")
    # else: # default to neutral

    # neutral only for now
    gender = "neutral"  # Ensure gender variable is set for return
    model_filename = "SMPL_NEUTRAL.pkl"
    logger.info(f"Loading SMPL model for neutral: {model_filename}")

    path = os.path.join(smpl_dir, model_filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"SMPL model file not found at {path}")

    body_model, _ = load_smpl_body_model(
        path=path,
        batch_size=1,
        num_betas=10,
        model_type=model_type,
        use_vtx_selector=True,
        device=device,
        fit_gender=gender,
        npz_hack=False,
    )
    return body_model, gender


def forward_smpl(body_model, trans, root_orient, body_pose, betas):
    preds = pred_smpl(
        body_model=body_model,
        trans=trans,
        root_orient=root_orient,
        body_pose=body_pose,
        betas=betas,
    )
    return preds


def reproject(intrinsics, R, t, key3d):
    """
    :param intrinsics: dict of 'fx','fy','cx','cy','k1','k2','k3'
    :param R rotation matrix from world to camera (T,3,3)
    :param t translation world to camera (T, 3)
    :param focal length (T, 2)
    :param principal point (T, 2)
    """
    n_frames, _ = t.shape
    device = t.device

    cam_center = torch.tensor(
        [intrinsics["cx"], intrinsics["cy"]], dtype=torch.float32, device=device
    ).reshape(1, 2)
    cam_f = torch.tensor(
        [intrinsics["fx"], intrinsics["fy"]], dtype=torch.float32, device=device
    ).reshape(1, 2)
    key3d = key3d.unsqueeze(0)  # only one body for now (T,N,3) -> (B, T, N, 3)
    R = R.unsqueeze(0)
    t = t.unsqueeze(0)

    # distortion
    if "k1" in intrinsics:
        cam_distortion_k = (
            torch.tensor(
                [intrinsics["k1"], intrinsics["k2"], intrinsics["k3"]],
                dtype=torch.float32,
                device=device,
            )
            .reshape(1, 3)
            .repeat(n_frames, 1)
        )
    else:
        cam_distortion_k = None
    if "p1" in intrinsics:
        cam_distortion_p = (
            torch.tensor(
                [intrinsics["p1"], intrinsics["p2"]],
                dtype=torch.float32,
                device=device,
            )
            .reshape(1, 2)
            .repeat(n_frames, 1)
        )
    else:
        cam_distortion_p = None

    key2d = camera.reproject(
        key3d,
        R,
        t,
        cam_f,
        cam_center,
        cam_distortion_k=cam_distortion_k,
        cam_distortion_p=cam_distortion_p,
    )
    """
    :param points3d (B, T, N, 3)
    :param cam_R (B, T, 3, 3)
    :param cam_t (B, T, 3)
    :param cam_f (T, 2)
    :param cam_center (T, 2)
    """

    return key2d


def getOpenPoseMarkerNames():
    markerNames = [
        "Nose",
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
        "midHip",
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

    return markerNames


def getOpenPoseFaceMarkers():
    faceMarkerNames = ["Nose", "REye", "LEye", "REar", "LEar"]
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]

    return faceMarkerNames, idxFaceMarkers


import numpy as np


def get_bounding_boxes(tensor, personframes_idx, numFrames, confidence_threshold=0.3):
    """
    Calculate bounding boxes for each frame based on keypoints with confidence above a threshold.

    Parameters:
        tensor (np.ndarray): Input tensor of shape (frames, keypoints, 3), where the last dimension
                             represents (x, y, confidence) for each keypoint.
        confidence_threshold (float): Confidence threshold for including a keypoint in bounding box calculation.

    Returns:
        np.ndarray: An array of shape (frames, 4), where each row is (min_x, min_y, max_x, max_y).
                    If no keypoints meet the threshold in a frame, the bounding box is set to (np.nan, np.nan, np.nan, np.nan).
    """
    # Initialize the bounding box array with NaNs
    bboxes = np.full((tensor.shape[0], 4), np.nan)
    bboxes_final = np.full((numFrames, 4), np.nan)

    # Loop through each frame
    for i, frame in enumerate(tensor):
        # Extract x, y, and confidence values
        x_coords = frame[:, 0]
        y_coords = frame[:, 1]
        confidences = frame[:, 2]

        # Mask for keypoints with confidence above the threshold
        valid_keypoints = confidences > confidence_threshold

        if np.any(valid_keypoints):  # Check if any keypoints meet the threshold
            # Calculate min and max for x and y using only the valid keypoints
            min_x = x_coords[valid_keypoints].min()
            max_x = x_coords[valid_keypoints].max()
            min_y_temp = y_coords[valid_keypoints].min()
            max_y_temp = y_coords[valid_keypoints].max()
            # enlarge the bounding box by 5% in the y direction
            min_y = min_y_temp - 0.05 * (max_y_temp - min_y_temp)
            max_y = max_y_temp + 0.05 * (max_y_temp - min_y_temp)

            # Store the bounding box in the result array
            bboxes[i] = [min_x, min_y, max_x, max_y]

    # fill in bbox_final with the bboxes using thr allPeopleframes_idx. The rest of the frames will be NaN
    bboxes_final[personframes_idx] = bboxes
    return bboxes_final


def keypointsToBoundingBox(data, confidenceThreshold=0.3):
    # output: nFrames x 4 (xTopLeft, yTopLeft, width, height).

    c_data = np.copy(data)

    remove_face_markers = False
    if remove_face_markers:
        # Remove face markers - they are intermittent.
        _, idxFaceMarkers = getOpenPoseFaceMarkers()
        idxToRemove = np.hstack([np.arange(i * 3, i * 3 + 3) for i in idxFaceMarkers])
        c_data = np.delete(c_data, idxToRemove, axis=1)

    # let's check the 3rd element of the 3rd dimension to see if the confidence of the keypoint is above the threshold
    # if it is, we keep the keypoint, otherwise we set it to nan
    c_data[c_data[:, :, 2] < confidenceThreshold] = np.nan

    bbox = np.zeros((c_data.shape[0], 4))
    # bbox[nonNanRows, 0] = np.nanmin(c_data[nonNanRows, 0::3], axis=2)

    # find all rows in which there is no nan value in the keypoints
    nonNanRows = np.all(~np.isnan(c_data), axis=2)
    # count the number of nonNanRows
    nNonNanRows = np.sum(nonNanRows, axis=1)

    # get the minimum x coordinate of the keypoints for each frame
    # bbox[:, 0] = np.nanmin(c_data[:, 0::3], axis=1)
    # # get the minimum y coordinate of the keypoints for each frame
    # bbox[:, 1] = np.nanmin(c_data[:, 1::3], axis=1)
    # # get the maximum x coordinate of the keypoints for each frame
    # bbox[:, 2] = (np.nanmax(c_data[:, 0::3], axis=1) - np.nanmin(c_data[:, 0::3], axis=1))
    # # get the maximum y coordinate of the keypoints for each frame
    # bbox[:, 3] = (np.nanmax(c_data[:, 1::3], axis=1) - np.nanmin(c_data[:, 1::3], axis=1))
    bbox[nonNanRows, 0] = np.nanmin(c_data[nonNanRows, 0::3], axis=1)
    bbox[nonNanRows, 1] = np.nanmin(c_data[nonNanRows, 1::3], axis=1)
    bbox[nonNanRows, 2] = np.nanmax(c_data[nonNanRows, 0::3], axis=1) - np.nanmin(
        c_data[nonNanRows, 0::3], axis=1
    )
    bbox[nonNanRows, 3] = np.nanmax(c_data[nonNanRows, 1::3], axis=1) - np.nanmin(
        c_data[nonNanRows, 1::3], axis=1
    )

    # Go a bit above head (this is for image-based tracker).
    bbox[:, 1] = np.maximum(0, bbox[:, 1] - 0.05 * bbox[:, 3])
    bbox[:, 3] = bbox[:, 3] * 1.05

    return bbox


def filter_keypoints_using_conf_rect(
    key2d, n_frames, frame_range_from_wham, print_stats=False
):
    key2d_np = key2d.cpu().numpy()
    last_frame = key2d_np.shape[0]
    conf_threshold = 0.8
    min_keypoints = 21

    # Compute statistics for each frame
    frame_stats = []
    for i_frame in range(key2d_np.shape[0]):
        confidences = key2d_np[i_frame, :, 2]
        above_threshold = confidences >= conf_threshold
        below_threshold = confidences < conf_threshold

        n_above = np.sum(above_threshold)
        n_below = np.sum(below_threshold)

        avg_conf_above = np.mean(confidences[above_threshold]) if n_above > 0 else 0.0
        avg_conf_below = np.mean(confidences[below_threshold]) if n_below > 0 else 0.0

        frame_stats.append(
            {
                "frame_idx": i_frame,
                "n_above_threshold": n_above,
                "avg_conf_above": avg_conf_above,
                "n_below_threshold": n_below,
                "avg_conf_below": avg_conf_below,
                "meets_criteria": n_above >= min_keypoints,
            }
        )

        # Print statistics if requested
        if print_stats:
            logger.info(
                f"Frame {i_frame}: {n_above} kpts >= {conf_threshold} (avg: {avg_conf_above:.3f}), "
                f"{n_below} kpts < {conf_threshold} (avg: {avg_conf_below:.3f}), "
                f"Meets criteria (>= {min_keypoints}): {n_above >= min_keypoints}"
            )

    high_confidence = key2d_np[:, :, 2] > conf_threshold
    # find the false indices in high_confidence
    # falses = np.where(~high_confidence)
    high_confidence = np.sum(high_confidence, axis=1)
    # there are 25 keypoints, filter out the frames having less than x keypoints with a confidence score higher than z
    high_confidence = high_confidence >= min_keypoints
    high_confidence_frames = np.where(high_confidence)
    if len(high_confidence_frames[0]) == 0:
        raise Exception("No frames with high confidence detected")
    lower_bound = high_confidence_frames[0][0]
    # Find the last frame on which at least 25 keypoints are detected with a confidence score higher than z
    upper_bound = high_confidence_frames[0][-1]
    # print(f"Frames with high confidence: {lower_bound } to {upper_bound}")
    first_frame_range = range(lower_bound, upper_bound)
    # print(f'Length of range: {len(first_frame_range)}')

    # for each frame, create a rectangle around the person using the leftmost and rightmost keypoints
    # and the topmost and bottommost keypoints
    keypoints_rect_coords = []

    lower_bound_rect_area = first_frame_range[0]
    upper_bound_rect_area = first_frame_range[-1]

    assert (
        lower_bound_rect_area < upper_bound_rect_area
    ), f"Lower bound of the rectangle area must be smaller than the upper bound {lower_bound_rect_area} !< {upper_bound_rect_area}"

    # Log summary statistics for tuning
    frames_meeting_criteria = sum(1 for stat in frame_stats if stat["meets_criteria"])
    total_frames = len(frame_stats)
    logger.info(
        f"Keypoint filtering summary: {frames_meeting_criteria}/{total_frames} frames meet criteria "
        f"(>= {min_keypoints} keypoints with confidence >= {conf_threshold})"
    )
    if frames_meeting_criteria > 0:
        avg_conf_above_all = np.mean(
            [
                stat["avg_conf_above"]
                for stat in frame_stats
                if stat["n_above_threshold"] > 0
            ]
        )
        avg_conf_below_all = np.mean(
            [
                stat["avg_conf_below"]
                for stat in frame_stats
                if stat["n_below_threshold"] > 0
            ]
        )
        logger.info(
            f"Average confidence - Above threshold: {avg_conf_above_all:.3f}, Below threshold: {avg_conf_below_all:.3f}"
        )

    opencap_mono_frame_range_wham_ref = range(
        lower_bound_rect_area, upper_bound_rect_area
    )

    first_detection_frame_id = frame_range_from_wham[0]
    last_detection_frame_id = frame_range_from_wham[-1]

    video_range = range(0, n_frames)
    opencap_mono_frame_range_video_ref = range(
        first_detection_frame_id + lower_bound_rect_area,
        first_detection_frame_id + upper_bound_rect_area,
    )
    wham_frame_range = range(first_detection_frame_id, last_detection_frame_id)

    return (
        video_range,
        wham_frame_range,
        opencap_mono_frame_range_wham_ref,
        opencap_mono_frame_range_video_ref,
        keypoints_rect_coords,
        frame_stats,  # Add frame statistics to return values
    )


def save_video_with_filters(
    cap,
    frame_rate,
    data_dir,
    opencap_mono_frame_range_video_ref,
    kpts_2d=None,
    opencap_mono_frame_range_wham_ref=None,
    original_video_path=None,
):
    """
    Save a debug video with frame annotations and keypoints.

    Args:
        cap: Video capture object
        frame_rate: Frame rate of the video
        data_dir: Directory to save the output video
        opencap_mono_frame_range_video_ref: Range of video frame numbers that have results
        kpts_2d: Optional keypoints tensor of shape (n_frames, 25, 3) corresponding to wham_ref frames
        opencap_mono_frame_range_wham_ref: Optional range of indices into kpts_2d (if provided, used to map kpts_2d to video frames)
        original_video_path: Optional path to original video (if different from cap)
    """
    if original_video_path is not None:
        print(f"original_video_path: {original_video_path}")
        cap, frame_rate, n_frames, _ = get_video_info(video_path=original_video_path)

    # Create mapping from video frame numbers to keypoint indices
    video_frame_to_kpt_idx = {}
    kpts_2d_np = None
    if kpts_2d is not None and opencap_mono_frame_range_wham_ref is not None:
        # Convert kpts_2d to numpy if it's a tensor
        if hasattr(kpts_2d, "cpu"):
            kpts_2d_np = kpts_2d.cpu().numpy()
        else:
            kpts_2d_np = kpts_2d

        # Map video frame numbers to keypoint array indices
        video_frames = list(opencap_mono_frame_range_video_ref)
        wham_indices = list(opencap_mono_frame_range_wham_ref)
        for vid_frame, wham_idx in zip(video_frames, wham_indices):
            if wham_idx < len(kpts_2d_np):
                video_frame_to_kpt_idx[vid_frame] = wham_idx

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(data_dir, "output_debug.mp4")
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(
            frame,
            str(frame_number),
            (50, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
        )

        if frame_number in opencap_mono_frame_range_video_ref:
            cv2.rectangle(frame, (50, 100), (210, 160), (0, 255, 0), -1)
            cv2.putText(
                frame,
                "OpenCap",
                (60, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        else:
            cv2.rectangle(frame, (50, 100), (150, 160), (0, 0, 255), -1)
            cv2.putText(
                frame,
                "None",
                (60, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Draw keypoints if available for this frame
        if frame_number in video_frame_to_kpt_idx and kpts_2d_np is not None:
            kpt_idx = video_frame_to_kpt_idx[frame_number]
            if kpt_idx < len(kpts_2d_np):
                for kpt in kpts_2d_np[kpt_idx]:
                    x, y, conf = kpt
                    if conf > 0.5:  # Only draw keypoints with confidence > 0.5
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    logger.info(f"Saved debug video to {output_path}")
    return output_path  # Return the path to the saved video


def findClosestBox(bbox, keyBoxes, imageSize, iPerson=None):
    # bbox: the bbox selected from the previous frame.
    # keyBoxes: bboxes detected in the current frame.
    # imageSize: size of the image
    # iPerson: index of the person to track..
    #
    # All boxes are xyxy: (min_x, min_y, max_x, max_y), same as get_bounding_boxes().

    # Parameters.
    # Proportion of mean image dimensions that corners must change to be
    # considered different person
    cornerChangeThreshold = 0.2

    keyBoxCorners = []
    for keyBox in keyBoxes:
        keyBoxCorners.append(np.array([keyBox[0], keyBox[1], keyBox[2], keyBox[3]]))
    bboxCorners = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

    boxErrors = [np.linalg.norm(keyBox - bboxCorners) for keyBox in keyBoxCorners]
    try:
        if iPerson is None:
            iPerson = np.nanargmin(boxErrors)
        bbox = keyBoxes[iPerson]
    except:
        return None, None, False

    # If large jump in bounding box, break.
    samePerson = True
    if boxErrors[iPerson] > cornerChangeThreshold * np.mean(imageSize):
        samePerson = False

    return iPerson, bbox, samePerson


def trackKeypointBox(
    videoPath,
    bbStart,
    allPeople,
    allBoxes,
    dataOut,
    frameStart=0,
    frameIncrement=1,
    visualize=False,
    badFramesBeforeStop=0,
):
    # Extract camera name
    if videoPath.split("InputMedia")[0][-5:-2] == "Cam":  # <= 10 cams
        camName = videoPath.split("InputMedia")[0][-5:-1]
    else:
        camName = videoPath.split("InputMedia")[0][-6:-1]

    # Tracks closest keypoint bounding boxes until the box changes too much.
    bboxKey = bbStart  # starting bounding box
    frameNum = frameStart

    # initiate video capture
    # Read video
    video = cv2.VideoCapture(videoPath.replace(".mov", "_rotated.avi"))
    nFrames = allBoxes[0].shape[0]

    # Read desiredFrames.
    video.set(1, frameNum)
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        raise Exception("Cannot read video file")

    imageSize = (frame.shape[0], frame.shape[1])
    justStarted = True
    count = 0
    badFrames = []
    while frameNum > -1 and frameNum < (nFrames - 1):
        # Read a new frame

        if visualize:
            video.set(1, frameNum)
            ok, frame = video.read()
            if not ok:
                break

        # Find person closest to tracked bounding box, and fill their keypoint data
        keyBoxes = [box[frameNum] for box in allBoxes]

        # get the boxes at the current frame for all people
        iPerson, bboxKey_new, samePerson = findClosestBox(bboxKey, keyBoxes, imageSize)

        # We allow badFramesBeforeStop of samePerson = False to account for an
        # errant frame(s) in the pose detector. Once we reach badFramesBeforeStop,
        # we break and output to the last good frame.
        if len(badFrames) > 0 and samePerson:
            badFrames = []

        if not samePerson and not justStarted:
            if len(badFrames) >= badFramesBeforeStop:
                print(
                    "{}: not same person at {}".format(
                        camName, frameNum - frameIncrement * badFramesBeforeStop
                    )
                )
                # Replace the data from the badFrames with zeros
                if len(badFrames) > 1:
                    dataOut[badFrames, :] = np.zeros(len(badFrames), dataOut.shape[0])
                break
            else:
                badFrames.append(frameNum)

        # Don't update the bboxKey for the badFrames
        if len(badFrames) == 0:
            bboxKey = bboxKey_new

        dataOut[frameNum, :] = allPeople[iPerson][frameNum, :]

        # Next frame
        frameNum += frameIncrement
        justStarted = False

        if visualize:
            p3 = (int(bboxKey[0]), int(bboxKey[1]))
            p4 = (int(bboxKey[2]), int(bboxKey[3]))
            cv2.rectangle(frame, p3, p4, (0, 255, 0), 2, 1)

            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        count += 1
    # find non zero values in dataOut
    nonZero = np.count_nonzero(dataOut, axis=1)
    if nonZero.size > 0:
        lastNonZero = np.max(np.where(nonZero > 0))
        dataOut[lastNonZero:, :] = 0
    return dataOut


def create_bbox_from_keypoints(keypoints):
    # Filter out invalid keypoints
    valid_keypoints = keypoints[keypoints[:, 2] > 0]
    if len(valid_keypoints) == 0:
        return None

    # clip the x and y to be positive
    valid_keypoints[:, 0] = np.clip(valid_keypoints[:, 0], 0, None)
    valid_keypoints[:, 1] = np.clip(valid_keypoints[:, 1], 0, None)

    # Calculate bbox (min x, min y, width, height)
    x_min, y_min = np.min(valid_keypoints[:, :2], axis=0)
    x_max, y_max = np.max(valid_keypoints[:, :2], axis=0)
    # enlarge the bounding box by 5%
    y_min = y_min - 0.05 * (y_max - y_min)
    y_max = y_max + 0.05 * (y_max - y_min)
    x_min = x_min - 0.05 * (x_max - x_min)
    x_max = x_max + 0.05 * (x_max - x_min)
    width, height = x_max - x_min, y_max - y_min
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def create_bbox_from_keypoints_centers(keypoints):
    x_min = keypoints[0]
    y_min = keypoints[1]
    x_max = keypoints[2]
    y_max = keypoints[3]
    width, height = x_max - x_min, y_max - y_min

    # Calculate the center of the bounding box
    center_x = x_min + width / 2
    center_y = y_min + height / 2

    # TODO double check this is the scale they use in wham
    scale = np.sqrt((width**2 + height**2)) / max(
        width, height
    )  # Normalized diagonal length

    # Return the center coordinates and scale
    return center_x, center_y, scale


def xyxy_to_cxyscale(keypoints):
    x_min = keypoints[0]
    y_min = keypoints[1]
    x_max = keypoints[2]
    y_max = keypoints[3]
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    scale = max(x_max - x_min, y_max - y_min) / 200
    return center_x, center_y, scale


def trackKeypointBox(
    videoPath,
    bbStart,
    allPeople,
    allBoxes,
    dataOut,
    frameStart=0,
    frameIncrement=1,
    visualize=False,
    badFramesBeforeStop=0,
):
    # Extract camera name
    if videoPath.split("InputMedia")[0][-5:-2] == "Cam":  # <= 10 cams
        camName = videoPath.split("InputMedia")[0][-5:-1]
    else:
        camName = videoPath.split("InputMedia")[0][-6:-1]

    # Tracks closest keypoint bounding boxes until the box changes too much.
    bboxKey = bbStart  # starting bounding box
    frameNum = frameStart

    # initiate video capture
    video = cv2.VideoCapture(videoPath.replace(".mov", "_rotated.avi"))
    nFrames = allBoxes[0].shape[0]

    # Read desired frames
    video.set(1, frameNum)
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        raise Exception("Cannot read video file")

    imageSize = (frame.shape[0], frame.shape[1])
    justStarted = True
    count = 0
    badFrames = []
    bboxOut = np.nan * np.zeros((nFrames, 3))
    bboxTest = np.nan * np.zeros((nFrames, 4))

    while frameNum > -1 and frameNum < (nFrames - 1):
        # Read a new frame
        if visualize:
            video.set(1, frameNum)
            ok, frame = video.read()
            if not ok:
                break

        # Get bounding boxes and find closest person
        keyBoxes = [box[frameNum] for box in allBoxes]
        iPerson, bboxKey_new, samePerson = findClosestBox(bboxKey, keyBoxes, imageSize)

        # Handle bad frames
        if len(badFrames) > 0 and samePerson:
            badFrames = []

        if not samePerson and not justStarted:
            if len(badFrames) >= badFramesBeforeStop:
                print(
                    "{}: not same person at {}".format(
                        camName, frameNum - frameIncrement * badFramesBeforeStop
                    )
                )
                if len(badFrames) > 1:
                    dataOut[badFrames, :] = np.zeros((len(badFrames), dataOut.shape[1]))
                break
            else:
                badFrames.append(frameNum)

        # Update bbox if no bad frames
        if len(badFrames) == 0:
            bboxKey = bboxKey_new

        # Store keypoints and bbox in dataOut
        keypoints = allPeople[iPerson][frameNum, :]
        dataOut[frameNum, :] = keypoints
        # Next frame
        frameNum += frameIncrement
        justStarted = False

        # Visualize
        if visualize:
            bbox_from_keypoints = create_bbox_from_keypoints(keypoints.reshape(-1, 3))
            if bbox_from_keypoints:
                p3 = (bbox_from_keypoints[0], bbox_from_keypoints[1])
                p4 = (bbox_from_keypoints[2], bbox_from_keypoints[3])
                cv2.rectangle(frame, p3, p4, (0, 255, 0), 2, 1)
                cv2.imshow("Tracking", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

        count += 1

    # Zero out trailing data in dataOut
    nonZero = np.count_nonzero(dataOut, axis=1)
    if nonZero.size > 0:
        lastNonZero = np.max(np.where(nonZero > 0))
        dataOut[lastNonZero:, :] = 0

    # create a bounding box for each frame using the keypoints
    for i in range(dataOut.shape[0]):
        bbox = create_bbox_from_keypoints(dataOut[i].reshape(-1, 3))
        if bbox:
            bbox_3 = xyxy_to_cxyscale(bbox)
            bboxOut[i] = bbox_3
        else:
            bboxOut[i, :] = np.nan
    return dataOut, bboxOut


def get_largest_bounding_box(all_data, all_bbox, conf_thresh=0.6):
    """
    Finds the largest bounding box for all people across all frames, focusing on high-confidence keypoints.

    Parameters:
        all_data (np.ndarray): Keypoint data for all people with shape (n_people, frames, keypoints, 3),
                               where each (frames, keypoints, 3) contains (x, y, confidence) for each person.
        all_bbox (np.ndarray): Bounding box data for all people with shape (n_people, frames, 4) as (min_x, min_y, max_x, max_y).
        conf_thresh (float): Minimum average confidence threshold for considering a bounding box.

    Returns:
        tuple: (max_area, max_idx, person_idx), where max_area is the largest box area (width × height in pixels),
               max_idx is the frame index, and person_idx is the index of the person with that bounding box.
    """
    # Parameters for filtering frames
    min_good_keypoints = 10
    low_conf_thresh = 0.4
    foot_conf_thresh = 0.5

    # Prepare data by replacing zeros with NaN
    c_data = np.where(all_data == 0, np.nan, all_data)
    conf = c_data[:, :, :, 2]
    c_bbox = np.copy(all_bbox)

    # Masks for filtering frames per person based on good keypoints and confidence
    enough_good_keypoints = (
        np.count_nonzero(~np.isnan(c_data), axis=(2, 3)) >= min_good_keypoints * 3
    )
    high_conf_keypoints = (
        np.count_nonzero(conf > low_conf_thresh, axis=2) >= min_good_keypoints
    )

    # Masks for foot confidence check per person
    marker_names = getOpenPoseMarkerNames()
    foot_indices = [
        marker_names.index(m)
        for m in ["RAnkle", "RHeel", "RBigToe", "LAnkle", "LHeel", "LBigToe"]
    ]
    foot_confidence = conf[:, :, foot_indices]
    # Suppress "Mean of empty slice" warning when computing mean on all-NaN slices
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        foot_mean = np.nanmean(foot_confidence, axis=2)
    foot_confidence_check = foot_mean >= foot_conf_thresh

    # Combine filters for each person
    valid_rows_mask = (
        enough_good_keypoints & high_conf_keypoints & foot_confidence_check
    )

    # Set invalid rows in bounding boxes to zero area
    c_bbox[~valid_rows_mask] = 0

    # Bounding boxes are xyxy (min_x, min_y, max_x, max_y) from get_bounding_boxes.
    width = c_bbox[:, :, 2] - c_bbox[:, :, 0]
    height = c_bbox[:, :, 3] - c_bbox[:, :, 1]
    bb_areas = np.maximum(width, 0.0) * np.maximum(height, 0.0)

    # Mask areas by rows with sufficient average confidence per person
    # Suppress "Mean of empty slice" warning when computing mean on all-NaN slices
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        avg_conf = np.nanmean(conf, axis=2)
    valid_conf_mask = avg_conf > conf_thresh
    masked_areas = bb_areas * valid_conf_mask

    # Find the largest bounding box area and corresponding indices
    max_area = np.nanmax(masked_areas)
    person_idx, max_idx = np.unravel_index(
        np.nanargmax(masked_areas), masked_areas.shape
    )

    return max_area, max_idx, person_idx


def rewriteVideos(
    inputPath,
    startFrame,
    nFrames,
    frameRate,
    outputDir=None,
    imageScaleFactor=0.5,
    outputFileName=None,
):

    inputDir, vidName = os.path.split(inputPath)
    vidName, vidExt = os.path.splitext(vidName)

    if outputFileName is None:
        outputFileName = vidName + "_sync" + vidExt
    if outputDir is not None:
        outputFullPath = os.path.join(outputDir, outputFileName)
    else:
        outputFullPath = os.path.join(inputDir, outputFileName)

    start_f = int(startFrame)
    end_f = int(startFrame) + int(nFrames) - 1
    if nFrames <= 0:
        logger.error("rewriteVideos: nFrames must be positive.")
        return outputFullPath

    # Frame-accurate trim: -ss before -i seeks to keyframes and desyncs kinematics (indexed by
    # frame_id) from the trimmed video. Use select on input frame index n, then CFR output at
    # frameRate so duration matches TRC/MOT (1/frameRate per row).
    vf_parts = [
        "select=between(n\\,{0}\\,{1})".format(start_f, end_f),
        "setpts=N/FRAME_RATE/TB",
    ]
    if imageScaleFactor is not None:
        vf_parts.append("scale=iw/{:.0f}:-1".format(1 / imageScaleFactor))
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        inputPath,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(float(frameRate)),
        "-frames:v",
        str(int(nFrames)),
        "-an",
        outputFullPath,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(
            "ffmpeg trim failed (code {}): {}\nCmd: {}",
            proc.returncode,
            (proc.stderr or "").strip(),
            " ".join(cmd[:6]) + " ... " + cmd[-1],
        )

    # logg the output file path
    logger.info(f"Successfully saved trimmed video (ffmpeg) to {outputFullPath}")

    return outputFullPath


def save_trimmed_video(
    data_dir: str,
    opencap_mono_frame_range_video_ref: range,
    original_video_path_or_folder: Optional[str] = None,  # Parameter name changed
    ffmpeg: bool = True,
) -> Optional[str]:
    """
    Saves a trimmed version of the video using ffmpeg or OpenCV, containing only frames within the
    opencap_mono_frame_range_video_ref.

    Args:
        data_dir: Directory to save the output trimmed video.
        opencap_mono_frame_range_video_ref: Range of frame numbers (video reference) to include.
        original_video_path_or_folder: Optional path to the original video file OR a folder containing it.
                                         If None, searches in data_dir.
        ffmpeg: If True, use ffmpeg (faster, requires install). If False, use OpenCV.

    Requires ffmpeg to be installed on the system if ffmpeg=True.

    Returns:
        str: Path to the saved trimmed video file, or None if saving failed.
    """
    input_video_path = None
    actual_frame_rate = None  # Will be determined from the video
    output_path = None  # Initialize output path

    # --- Determine input_video_path ---
    if original_video_path_or_folder is not None:
        if os.path.isfile(original_video_path_or_folder):
            input_video_path = original_video_path_or_folder
            logger.info(
                f"Using provided direct video file for trimming: {input_video_path}"
            )
        elif os.path.isdir(original_video_path_or_folder):
            logger.info(
                f"Searching for video in provided folder for trimming: {original_video_path_or_folder}"
            )
            # Standardize video file finding
            found_videos = [
                f
                for f in os.listdir(original_video_path_or_folder)
                if (f.endswith((".mp4", ".avi", ".mov")))
                and "sync" not in f.lower()
                and "debug" not in f.lower()
                and "trimmed" not in f.lower()
            ]
            if found_videos:
                # Sort to get consistent results if multiple videos are present
                found_videos.sort()
                input_video_path = os.path.join(
                    original_video_path_or_folder, found_videos[0]
                )
                logger.info(f"Found video in folder for trimming: {input_video_path}")
            else:
                logger.warning(
                    f"No suitable video file found in provided folder for trimming: {original_video_path_or_folder}."
                )
        else:
            logger.warning(
                f"Provided 'original_video_path_or_folder' is not a valid file or directory: {original_video_path_or_folder}"
            )

    # Fallback to data_dir if no input_video_path determined yet
    if input_video_path is None:
        logger.info(
            f"No valid 'original_video_path_or_folder' provided or video not found, searching in 'data_dir' for trimming: {data_dir}"
        )
        try:
            found_videos = [
                f
                for f in os.listdir(data_dir)
                if (f.endswith((".mp4", ".avi", ".mov")))
                and "sync" not in f.lower()
                and "debug" not in f.lower()
                and "trimmed" not in f.lower()
            ]
            if found_videos:
                # Sort to get consistent results if multiple videos are present
                found_videos.sort()
                input_video_path = os.path.join(data_dir, found_videos[0])
                logger.info(
                    f"Found video in 'data_dir' for trimming: {input_video_path}"
                )
            else:
                logger.error(
                    f"No suitable input video found for trimming in 'data_dir': {data_dir}."
                )
                return None
        except FileNotFoundError:
            logger.error(f"Cannot search for video: data_dir '{data_dir}' not found.")
            return None
        except Exception as e:
            logger.error(f"Error searching for video in data_dir '{data_dir}': {e}")
            return None

    if not input_video_path or not os.path.exists(input_video_path):
        logger.error(
            f"Final input video path for trimming is invalid or not found: {input_video_path}"
        )
        return None

    # --- Get frame rate from the determined input_video_path ---
    try:
        _, actual_frame_rate, _, _ = get_video_info(
            video_path=input_video_path, release=True
        )
        if actual_frame_rate is None or actual_frame_rate <= 0:
            raise ValueError("Frame rate from video info is invalid (<= 0 or None).")
        logger.info(
            f"Determined frame rate for trimming: {actual_frame_rate} from {input_video_path}"
        )
    except Exception as e:
        logger.error(
            f"Could not get valid frame rate from video '{input_video_path}'. Error: {e}"
        )
        return None

    # Check if frame range is valid
    if not opencap_mono_frame_range_video_ref:
        logger.error("opencap_mono_frame_range_video_ref is empty. Cannot trim video.")
        return None

    min_frame = min(opencap_mono_frame_range_video_ref)
    max_frame = max(opencap_mono_frame_range_video_ref)

    # --- Perform Trimming ---
    if ffmpeg:
        logger.info(f"Attempting to trim video using ffmpeg: {input_video_path}")

        # output_path = os.path.join(data_dir, "output_trimmed.mp4")
        num_frames = max_frame - min_frame + 1

        output_path = rewriteVideos(
            inputPath=input_video_path,
            startFrame=min_frame,
            nFrames=num_frames,
            frameRate=actual_frame_rate,
            outputDir=data_dir,
            outputFileName="output_trimmed.mp4",
        )

    else:
        # Use OpenCV to trim the video
        logger.info(f"Attempting to trim video using OpenCV: {input_video_path}")
        cap_cv = None
        out_cv = None
        try:
            cap_cv = cv2.VideoCapture(input_video_path)
            if not cap_cv.isOpened():
                logger.error(f"OpenCV could not open video file: {input_video_path}")
                return None  # Return None as output_path is still None

            width = int(cap_cv.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_cv.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Use MP4V codec for .mp4 output
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = os.path.join(data_dir, "output_trimmed_opencv.mp4")
            out_cv = cv2.VideoWriter(
                output_path, fourcc, actual_frame_rate, (width, height)
            )

            if not out_cv.isOpened():
                logger.error(
                    f"OpenCV could not open VideoWriter for output path: {output_path}"
                )
                output_path = None  # Ensure None is returned
                return None

            # Set the starting frame for reading
            cap_cv.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

            current_frame_num = min_frame
            while cap_cv.isOpened():
                # Check if we have processed all required frames
                if current_frame_num > max_frame:
                    break

                ret, frame = cap_cv.read()

                if not ret:
                    logger.warning(
                        f"OpenCV could not read frame {current_frame_num} (or end of video reached unexpectedly). Stopping trim."
                    )
                    break

                # Write the frame if it's within the desired range
                out_cv.write(frame)

                current_frame_num += 1

            logger.info(f"Successfully saved trimmed video (OpenCV) to {output_path}")

        except Exception as e:
            logger.error(f"An error occurred during OpenCV video trimming: {e}")
            output_path = None  # Ensure None is returned on error
        finally:
            # Ensure OpenCV resources are released
            if cap_cv is not None and cap_cv.isOpened():
                cap_cv.release()
            if (
                out_cv is not None and out_cv.isOpened()
            ):  # Check if out_cv was successfully created before releasing
                out_cv.release()

    return output_path
