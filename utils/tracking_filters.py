import os.path as osp
import json
import numpy as np
from loguru import logger


class InsufficientFullBodyKeypointsError(ValueError):
    """Raised when no frame meets the minimum full-body 2D keypoint confidence rule."""

    pass


def handle_multi_person_tracking(
    tracking_results, video, num_kpts, nFrames, output_pth
):
    """
    Handle multi-person detection and select the person of interest.

    Args:
        tracking_results: Dictionary of tracking results for all detected people
        video: Video path
        num_kpts: Number of keypoints
        nFrames: Total number of frames
        output_pth: Output path for saving intermediate results

    Returns:
        tracking_results: Updated tracking results with only person of interest (key=0)
    """
    sub_nums = []
    Ts = []
    allPeoplePose = []
    allPeopleframes_idx = {}

    for i_subject, values in tracking_results.items():
        frame_ids = np.asarray(values["frame_id"]).astype(int)
        keypoints = np.asarray(values["keypoints"])

        # Keep frame_id and keypoints aligned even if upstream outputs mismatch.
        if len(frame_ids) != len(keypoints):
            min_len = min(len(frame_ids), len(keypoints))
            logger.warning(
                f"Tracking length mismatch for subject {i_subject}: "
                f"{len(frame_ids)} frame_ids vs {len(keypoints)} keypoints. "
                f"Truncating to {min_len}."
            )
            frame_ids = frame_ids[:min_len]
            keypoints = keypoints[:min_len]

        # Guard against off-by-one/invalid indices from detector output.
        valid_mask = (frame_ids >= 0) & (frame_ids < nFrames)
        if not np.all(valid_mask):
            n_removed = int((~valid_mask).sum())
            logger.warning(
                f"Removed {n_removed} out-of-range frame_ids for subject {i_subject} "
                f"(valid range: 0..{nFrames - 1})."
            )
            frame_ids = frame_ids[valid_mask]
            keypoints = keypoints[valid_mask]

        sub_nums.append(i_subject)
        Ts.append(len(keypoints))
        allPeoplePose.append(keypoints)
        allPeopleframes_idx[i_subject] = frame_ids

    allPeopleframes_idx_str_keys = {
        str(key): value for key, value in allPeopleframes_idx.items()
    }
    # convert the arrays to list which are the values in the dict
    allPeopleframes_idx_str_keys = {
        key: value.tolist() for key, value in allPeopleframes_idx_str_keys.items()
    }

    # save the allPeopleframes_idx to a json file
    allPeopleframes_idx_path = osp.join(output_pth, "allPeopleframes_idx.json")
    with open(allPeopleframes_idx_path, "w") as f:
        json.dump(allPeopleframes_idx_str_keys, f)

    # number of subjects appearing in the video (can be the same person multiple times if coming in and out of the frames)
    num_people = len(tracking_results.items())

    if num_people > 1:
        logger.info(f"More than one person detected in the video ({num_people}).")
        from utils.utils_optim import (
            get_largest_bounding_box,
            trackKeypointBox,
            get_bounding_boxes,
        )

        confidenceThresholdForBB = 0.3
        # Select the largest keypoint-based bounding box as the subject of interest
        bbFromKeypoints = []
        for data in allPeoplePose:
            idx = next(
                i for i, item in enumerate(allPeoplePose) if np.array_equal(item, data)
            )
            personframes_idx = allPeopleframes_idx[sub_nums[idx]]
            # is this necessary?
            # we should just use the bounding boxes from the tracking results
            bbFromKeypoints.append(
                get_bounding_boxes(
                    data,
                    personframes_idx,
                    nFrames,
                    confidence_threshold=confidenceThresholdForBB,
                )
            )

        # Main block
        all_data_with_nan = [
            np.full((nFrames, num_kpts, 3), np.nan) for _ in range(len(allPeoplePose))
        ]
        for i, data in enumerate(allPeoplePose):
            personframes_idx = allPeopleframes_idx[sub_nums[i]]
            all_data_with_nan[i][personframes_idx] = data

        all_data_with_nan = np.stack(
            all_data_with_nan
        )  # Shape (n_people, nFrames, 25, 3)
        all_data_with_nan_list = [
            np.full((nFrames, num_kpts, 3), np.nan) for _ in range(len(allPeoplePose))
        ]
        # all_data_with_nan to list
        for i, data in enumerate(allPeoplePose):
            personframes_idx = allPeopleframes_idx[sub_nums[i]]
            all_data_with_nan_list[i][personframes_idx] = data
        all_bbox = np.stack(bbFromKeypoints)  # Shape (n_people, nFrames, 4)

        max_area, max_idx, person_idx = get_largest_bounding_box(
            all_data_with_nan, all_bbox, conf_thresh=confidenceThresholdForBB
        )

        if max_area == 0.0:
            key2D = np.zeros((num_kpts, nFrames, 2))
            confidence = np.zeros((num_kpts, nFrames))
            return key2D, confidence

        # Get the starting keypoints and bounding box for the detected person
        startFrame = max_idx
        startBb = all_bbox[person_idx, startFrame]

        # initialize output data
        res = np.zeros((nFrames, num_kpts, 3))

        # track this bounding box backwards until it can't be tracked
        res, res_bbox = trackKeypointBox(
            videoPath=video,
            bbStart=startBb,
            allPeople=all_data_with_nan_list,
            allBoxes=bbFromKeypoints,
            dataOut=res,
            frameStart=startFrame,
            frameIncrement=-1,
            visualize=False,
        )

        # track this bounding box forwards until it can't be tracked
        res, res_bbox = trackKeypointBox(
            videoPath=video,
            bbStart=startBb,
            allPeople=all_data_with_nan_list,
            allBoxes=bbFromKeypoints,
            dataOut=res,
            frameStart=startFrame,
            frameIncrement=1,
            visualize=False,
        )

        indices_poi = np.where(np.any(res != 0, axis=(1, 2)))[0]
        keypoints_poi = res[indices_poi]
        # get the bbox at the indices of the person of interest
        bbox_poi = res_bbox[indices_poi]

        tracking_results = {
            0: {
                "features": [],
                "frame_id": indices_poi,
                "bbox": bbox_poi,
                "keypoints": keypoints_poi,
            }
        }
    else:
        logger.info(f"Only one person detected in the video.")
        # if the key of the person in tracking_results is not 0, then change it to 0
        # add a check in case there are no keys in tracking_results
        if len(tracking_results) == 0:
            raise ValueError("No tracking results found")
        # add a check in case there are no keys in tracking_results
        if list(tracking_results.keys())[0] != 0:
            tracking_results[0] = tracking_results.pop(list(tracking_results.keys())[0])

    return tracking_results


def filter_frames_by_bbox_height(
    tracking_results, height, width, min_ratio=0.15, max_ratio=0.95
):
    """
    Filter start and end frames based on bbox height relative to video dimensions.

    Args:
        tracking_results: Dictionary of tracking results (modified in place)
        height: Video height
        width: Video width
        min_ratio: Minimum bbox height ratio (default: 0.15 = 15%)
        max_ratio: Maximum bbox height ratio (default: 0.95 = 95%)

    Returns:
        tracking_results: Dictionary of tracking results with filtered frames
    """
    if len(tracking_results) == 0 or 0 not in tracking_results:
        return

    poi_data = tracking_results[0]
    keypoints_poi = poi_data["keypoints"]
    frame_ids = poi_data["frame_id"]

    if len(keypoints_poi) == 0:
        return

    # Use the larger dimension (handles both portrait and landscape videos)
    # In landscape, if human is vertical, compare bbox height with video width
    max_dimension = max(height, width)

    # Compute bbox height for each frame from keypoints
    bbox_heights = []
    for kp_frame in keypoints_poi:
        # Filter valid keypoints (confidence > 0)
        valid_kp = kp_frame[kp_frame[:, 2] > 0]
        if len(valid_kp) > 0:
            y_min = np.min(valid_kp[:, 1])
            y_max = np.max(valid_kp[:, 1])
            # Apply 5% enlargement like in create_bbox_from_keypoints
            bbox_height = (y_max - y_min) * 1.1  # 5% on each side = 10% total
        else:
            bbox_height = 0
        bbox_heights.append(bbox_height)

    bbox_heights = np.array(bbox_heights)
    # Compute height as percentage of max dimension (height or width, whichever is larger)
    height_ratios = bbox_heights / max_dimension

    # Find valid frames (height between min_ratio and max_ratio of max dimension)
    valid_mask = (height_ratios >= min_ratio) & (height_ratios <= max_ratio)

    # Find first valid frame from start
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        start_valid_idx = valid_indices[0]
        end_valid_idx = valid_indices[-1]

        # Identify which frames were removed
        removed_start_frames = []
        if start_valid_idx > 0:
            removed_start_frames = frame_ids[:start_valid_idx].tolist()

        removed_end_frames = []
        if end_valid_idx < len(frame_ids) - 1:
            removed_end_frames = frame_ids[end_valid_idx + 1 :].tolist()

        # Apply the filter (only keep frames from start_valid_idx to end_valid_idx)
        filtered_indices = np.arange(start_valid_idx, end_valid_idx + 1)
        tracking_results[0]["frame_id"] = frame_ids[filtered_indices]
        tracking_results[0]["keypoints"] = keypoints_poi[filtered_indices]
        if "bbox" in tracking_results[0] and len(tracking_results[0]["bbox"]) > 0:
            tracking_results[0]["bbox"] = tracking_results[0]["bbox"][filtered_indices]

        # Log the results
        total_removed = len(removed_start_frames) + len(removed_end_frames)
        if total_removed > 0:
            logger.info(
                f"Filtered frames: kept {len(filtered_indices)}/{len(frame_ids)} frames "
                f"(removed {len(removed_start_frames)} from start, {len(removed_end_frames)} from end)"
            )
            if len(removed_start_frames) > 0:
                logger.info(f"Removed start frames: {removed_start_frames}")
            if len(removed_end_frames) > 0:
                logger.info(f"Removed end frames: {removed_end_frames}")
        else:
            logger.info(
                f"No frames removed. All {len(frame_ids)} frames have valid bbox height ({min_ratio*100:.0f}%-{max_ratio*100:.0f}% of max dimension)."
            )
    else:
        logger.warning(
            f"No frames with valid bbox height ({min_ratio*100:.0f}%-{max_ratio*100:.0f}% of max dimension) found. Keeping all frames."
        )
    return tracking_results


def filter_frames_by_bbox_touching_edges(tracking_results, height, width, margin=1):
    """
    Trim start/end frames where the person's bbox touches any image edge.

    Args:
        tracking_results: Dictionary of tracking results (modified in place)
        height: Video height
        width: Video width
        margin: Pixel margin for "touching" an edge (default: 1)

    Returns:
        tracking_results: Dictionary of tracking results with filtered frames
    """
    if len(tracking_results) == 0 or 0 not in tracking_results:
        return tracking_results

    poi_data = tracking_results[0]
    keypoints_poi = poi_data["keypoints"]
    frame_ids = poi_data["frame_id"]

    if len(keypoints_poi) == 0:
        logger.warning("No keypoints found in tracking results.")
        return tracking_results

    touching_mask = []
    for kp_frame in keypoints_poi:
        valid_kp = kp_frame[kp_frame[:, 2] > 0]
        if len(valid_kp) == 0:
            touching_mask.append(True)
            continue

        # Clip negative coords then compute bbox
        x = np.clip(valid_kp[:, 0], 0, None)
        y = np.clip(valid_kp[:, 1], 0, None)
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        # Apply 5% enlargement like create_bbox_from_keypoints
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min = x_min - 0.05 * x_range
        x_max = x_max + 0.05 * x_range
        y_min = y_min - 0.05 * y_range
        y_max = y_max + 0.05 * y_range

        touches = (
            (x_min <= margin)
            or (y_min <= margin)
            or (x_max >= (width - 1 - margin))
            or (y_max >= (height - 1 - margin))
        )
        touching_mask.append(touches)

    touching_mask = np.array(touching_mask)
    valid_indices = np.where(~touching_mask)[0]

    if len(valid_indices) == 0:
        logger.warning(
            "No frames found with bbox fully inside the image. Keeping all frames."
        )
        return tracking_results

    start_valid_idx = valid_indices[0]
    end_valid_idx = valid_indices[-1]

    removed_start_frames = []
    if start_valid_idx > 0:
        removed_start_frames = frame_ids[:start_valid_idx].tolist()

    removed_end_frames = []
    if end_valid_idx < len(frame_ids) - 1:
        removed_end_frames = frame_ids[end_valid_idx + 1 :].tolist()

    filtered_indices = np.arange(start_valid_idx, end_valid_idx + 1)
    tracking_results[0]["frame_id"] = frame_ids[filtered_indices]
    tracking_results[0]["keypoints"] = keypoints_poi[filtered_indices]
    if "bbox" in tracking_results[0] and len(tracking_results[0]["bbox"]) > 0:
        tracking_results[0]["bbox"] = tracking_results[0]["bbox"][filtered_indices]

    total_removed = len(removed_start_frames) + len(removed_end_frames)
    if total_removed > 0:
        logger.info(
            f"Filtered frames by bbox edges: kept {len(filtered_indices)}/{len(frame_ids)} frames "
            f"(removed {len(removed_start_frames)} from start, {len(removed_end_frames)} from end)"
        )
        if len(removed_start_frames) > 0:
            logger.info(f"Removed start frames: {removed_start_frames}")
        if len(removed_end_frames) > 0:
            logger.info(f"Removed end frames: {removed_end_frames}")
    else:
        logger.info(
            f"No frames removed. All {len(frame_ids)} frames have bbox fully inside the image."
        )

    return tracking_results


def filter_frames_by_keypoints(tracking_results, conf_threshold=0.8, min_keypoints=21):
    """
    Filter frames based on keypoint confidence.
    Converts 133 keypoints to 25 OpenPose keypoints, then filters frames where
    at least min_keypoints have confidence >= conf_threshold.

    Args:
        tracking_results: Dictionary of tracking results (modified in place)
        conf_threshold: Confidence threshold for keypoints (default: 0.8)
        min_keypoints: Minimum number of keypoints with high confidence (default: 21)

    Returns:
        tracking_results: Dictionary of tracking results with filtered frames

    Raises:
        InsufficientFullBodyKeypointsError: If no frame meets the confidence rule.
    """
    if len(tracking_results) == 0 or 0 not in tracking_results:
        return tracking_results

    poi_data = tracking_results[0]
    keypoints_original = poi_data[
        "keypoints"
    ]  # Keep original keypoints (133) - don't modify
    frame_ids = poi_data["frame_id"]

    if len(keypoints_original) == 0:
        logger.warning("No keypoints found in tracking results.")
        return tracking_results

    # Convert to numpy if tensor (like in filter_keypoints_using_conf_rect) - for filtering only
    if hasattr(keypoints_original, "cpu"):
        keypoints_poi_np = keypoints_original.cpu().numpy()
    elif hasattr(keypoints_original, "numpy"):
        keypoints_poi_np = keypoints_original.numpy()
    else:
        keypoints_poi_np = np.asarray(keypoints_original)

    # For filtering, convert from 133 to 25 OpenPose keypoints (like in filter_keypoints_using_conf_rect)
    # We use 25 keypoints only for filtering decisions, but keep original 133 in tracking_results
    if keypoints_poi_np.shape[1] == 133:
        # Import here to avoid circular imports
        from utils.utils_optim import get_openpose_keypoints

        # Convert 133 -> 25 OpenPose keypoints temporarily for filtering (get_openpose_keypoints expects numpy, returns torch)
        keypoints_25 = get_openpose_keypoints(keypoints_poi_np, device="cpu")
        # Convert back to numpy for filtering
        if hasattr(keypoints_25, "cpu"):
            keypoints_for_filtering = keypoints_25.cpu().numpy()
        else:
            keypoints_for_filtering = np.asarray(keypoints_25)
    elif keypoints_poi_np.shape[1] == 25:
        # Already 25 keypoints, use as is for filtering
        keypoints_for_filtering = keypoints_poi_np
    else:
        # Other number of keypoints (e.g., 17), use as is for filtering
        keypoints_for_filtering = keypoints_poi_np
        logger.info(
            f"Using {keypoints_poi_np.shape[1]} keypoints for filtering (expected 133 or 25)"
        )

    # Ensure conf_threshold is a float, not a tensor
    if hasattr(conf_threshold, "item"):
        conf_threshold = conf_threshold.item()
    conf_threshold = float(conf_threshold)

    # Count high confidence keypoints per frame using the keypoints for filtering
    # keypoints shape: (n_frames, n_keypoints, 3) where last dim is [x, y, confidence]
    confidences = keypoints_for_filtering[:, :, 2]  # Shape: (n_frames, n_keypoints)
    high_conf_mask = confidences >= conf_threshold
    n_high_conf_per_frame = np.sum(high_conf_mask, axis=1)  # Shape: (n_frames,)

    n_frames = int(n_high_conf_per_frame.shape[0])
    n_kpts_filter = keypoints_for_filtering.shape[1]
    nh = np.asarray(n_high_conf_per_frame, dtype=np.float64)
    below_min = int(np.sum(nh < min_keypoints))
    zero_high = int(np.sum(nh == 0))

    logger.info(
        f"[keypoint_filter] input: {n_frames} frames, {n_kpts_filter} kpts used for rule, "
        f"threshold>={conf_threshold}, require>={min_keypoints} high-conf kpts/frame"
    )
    logger.info(
        f"[keypoint_filter] high-conf count per frame: min={int(nh.min())}, max={int(nh.max())}, "
        f"mean={nh.mean():.2f}, median={float(np.median(nh)):.1f}"
    )
    logger.info(
        f"[keypoint_filter] frames with count < {min_keypoints}: {below_min}/{n_frames}; "
        f"frames with 0 high-conf: {zero_high}/{n_frames}"
    )

    # Sample a few frames (index -> frame_id -> count) for debugging partial body / hand-only clips
    frame_ids_np = np.asarray(frame_ids)
    sample_idx = sorted(
        set(
            list(range(min(3, n_frames)))
            + list(range(max(0, n_frames // 2 - 1), min(n_frames, n_frames // 2 + 2)))
            + list(range(max(0, n_frames - 3), n_frames))
        )
    )
    sample_pairs = []
    for si in sample_idx:
        fid = int(frame_ids_np[si]) if si < len(frame_ids_np) else si
        sample_pairs.append(f"fid={fid}:{int(nh[si])}")
    logger.info(f"[keypoint_filter] sample n_high_conf (>={conf_threshold}): " + "; ".join(sample_pairs))

    # Find frames meeting criteria
    valid_mask = n_high_conf_per_frame >= min_keypoints
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        msg = (
            f"No frame has >= {min_keypoints} keypoints at conf >= {conf_threshold} "
            f"(hand-only / partial body / bad visibility). Evaluated {n_frames} frames; "
            f"min/mean/max high-conf counts: {int(nh.min())}/{nh.mean():.2f}/{int(nh.max())}"
        )
        logger.warning(f"[keypoint_filter] {msg} — aborting WHAM preprocess.")
        raise InsufficientFullBodyKeypointsError(msg)

    # Find first and last valid frame
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]

    # Apply the filter (only keep frames from first_valid_idx to last_valid_idx)
    # Use original keypoints (133) - just trim frames, don't modify keypoints
    filtered_indices = np.arange(first_valid_idx, last_valid_idx + 1)
    tracking_results[0]["frame_id"] = frame_ids[filtered_indices]

    # Keep original keypoints (133) - just filter by frame indices (works for both tensor and numpy)
    tracking_results[0]["keypoints"] = keypoints_original[filtered_indices]

    if "bbox" in tracking_results[0] and len(tracking_results[0]["bbox"]) > 0:
        tracking_results[0]["bbox"] = tracking_results[0]["bbox"][filtered_indices]

    # Log the results
    total_removed = len(frame_ids) - len(filtered_indices)
    if total_removed > 0:
        logger.info(
            f"Filtered frames by keypoints: kept {len(filtered_indices)}/{len(frame_ids)} frames "
            f"(removed {total_removed} frames with < {min_keypoints} keypoints at confidence >= {conf_threshold})"
        )
    else:
        logger.info(
            f"No frames removed. All {len(frame_ids)} frames have >= {min_keypoints} keypoints at confidence >= {conf_threshold}."
        )

    return tracking_results
