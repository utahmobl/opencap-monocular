import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import argparse
import cv2
from loguru import logger
from pathlib import Path

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.utils_optim as ut

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.utils_optim as ut


def visualize_contact_probabilities(
    pkl_path=None,
    data_dir=None,
    output_dir=None,
    before_threshold=True,
    after_threshold=True,
    false_true_thresh=0.5,
    true_false_thresh=0.5,
    use_seconds=True,
    min_stretch_len=1,
):
    """
    Visualize feet contact probabilities for a video.

    Args:
        pkl_path (str): Direct path to the WHAM output.pkl file (if provided, data_dir is ignored)
        data_dir (str): Directory containing the WHAM output.pkl file (only used if pkl_path is None)
        output_dir (str): Directory to save the plots (defaults to containing directory of the pkl file)
        before_threshold (bool): Plot the contact probabilities before thresholding
        after_threshold (bool): Plot the contact probabilities after thresholding
        false_true_thresh (float): Threshold value for switching from non-contact to contact
        true_false_thresh (float): Threshold value for switching from contact to non-contact
        use_seconds (bool): If True, show time in seconds on x-axis, otherwise show frame numbers
        min_stretch_len (int): Minimum stretch length for debounced threshold (default: 1)
    """
    # Handle direct PKL path or data directory
    if pkl_path is not None:
        wham_regression_results_pth = pkl_path
        data_dir = os.path.dirname(pkl_path)
    elif data_dir is not None:
        wham_regression_results_pth = os.path.join(data_dir, "wham_output.pkl")
    else:
        logger.error("Either pkl_path or data_dir must be provided")
        return

    if output_dir is None:
        output_dir = data_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load WHAM results
    try:
        wham_regression_results = ut.load_results(wham_regression_results_pth)
        logger.info(f"Loaded WHAM results from {wham_regression_results_pth}")
    except FileNotFoundError:
        logger.error(f"Could not find WHAM results at {wham_regression_results_pth}")
        return
    except Exception as e:
        logger.error(f"Error loading WHAM results: {e}")
        return

    # Load video and get frame rate
    try:
        video_dir = data_dir
        _, frame_rate, n_frames, _ = ut.get_video_info(data_dir=video_dir, release=True)
        logger.info(f"Frame rate: {frame_rate}")
    except Exception as e:
        logger.warning(f"Could not determine frame rate, using default 30 fps: {e}")
        frame_rate = 30

    # Process WHAM results - handle multiple people if detected
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

    # Get contact information
    contact = wham_result["contact"]

    # Get original frame IDs from WHAM output
    if "frame_ids" in wham_result:
        frame_ids = wham_result["frame_ids"]
        logger.info(
            f"Using original frame IDs from WHAM output, shape: {frame_ids.shape}"
        )
    elif "frame_id" in wham_result:
        frame_ids = wham_result["frame_id"]
        logger.info(
            f"Using original frame ID from WHAM output, shape: {frame_ids.shape}"
        )
    else:
        logger.warning("No frame IDs found in WHAM output, creating frame index array")
        frame_ids = np.arange(len(contact))

    # Create x-axis values (time in seconds or frame numbers)
    if use_seconds:
        x_values = frame_ids / frame_rate
        x_label = "Time (seconds)"
    else:
        x_values = frame_ids
        x_label = "Frame Number"

    # Apply Butterworth filter to smooth the contact probabilities
    filtered_contact = ut.filter_array(
        contact, order=4, cutoff_freq=6, sampling_rate=frame_rate
    )

    # Apply threshold (debounced) to get binary contact mask
    contact_tensor = torch.from_numpy(filtered_contact)
    debounced_contacts = debounced_threshold(
        contact_tensor,
        false_true_thresh_heel=false_true_thresh,
        true_false_thresh_heel=true_false_thresh,
        false_true_thresh_big_toe=false_true_thresh,
        true_false_thresh_big_toe=true_false_thresh,
        min_stretch_len=min_stretch_len,
    )

    # Get names of the contact points (feet)
    foot_names = ["Left Big Toe", "Left Heel", "Right Big Toe", "Right Heel"]

    # Create a figure for each contact point
    for i in range(contact.shape[1]):
        plt.figure(figsize=(12, 6))

        # Plot original contact probabilities if requested
        if before_threshold:
            plt.plot(
                x_values,
                contact[:, i],
                label="Original Contact",
                color="blue",
                alpha=0.5,
            )

        # Plot filtered contact probabilities
        plt.plot(x_values, filtered_contact[:, i], label="Filtered", color="green")

        # Plot the thresholded (binary) contacts if requested
        if after_threshold:
            plt.plot(
                x_values,
                debounced_contacts.numpy()[:, i].astype(float),
                label="Thresholded",
                color="red",
                linestyle="--",
            )

        # Add threshold lines
        plt.axhline(
            y=false_true_thresh,
            color="purple",
            linestyle=":",
            label=f"False to true threshold ({false_true_thresh})",
        )
        plt.axhline(
            y=true_false_thresh,
            color="orange",
            linestyle=":",
            label=f"True to false threshold ({true_false_thresh})",
        )

        # Add more x-axis ticks for better precision
        plt.xticks(np.linspace(min(x_values), max(x_values), num=20, endpoint=True))

        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel("Contact Probability")
        plt.title(
            f"Contact Probabilities for {foot_names[i]}\n(Requires 3 consecutive frames above 'False to true' or below 'True to false' thresholds to change state)"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Generate output filename
        # Use either the trial name from the directory or the PKL filename
        if pkl_path:
            trial_name = os.path.splitext(os.path.basename(pkl_path))[0]
        else:
            trial_name = os.path.basename(data_dir)

        output_path = os.path.join(
            output_dir,
            f"contact_probabilities_{trial_name}_{foot_names[i].replace(' ', '_')}.png",
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved contact plot to {output_path}")

        plt.close()

    # Create a combined plot with all contact points
    plt.figure(figsize=(12, 8))

    # Create subplots for each contact point
    for i in range(contact.shape[1]):
        plt.subplot(contact.shape[1], 1, i + 1)

        # Plot filtered contact probabilities
        plt.plot(x_values, filtered_contact[:, i], label="Filtered", color="green")

        # Plot the thresholded (binary) contacts if requested
        if after_threshold:
            plt.plot(
                x_values,
                debounced_contacts.numpy()[:, i].astype(float),
                label="Thresholded",
                color="red",
                linestyle="--",
            )

        # Add threshold lines
        plt.axhline(
            y=false_true_thresh,
            color="purple",
            linestyle=":",
            label=f"False to true threshold ({false_true_thresh})",
        )
        plt.axhline(
            y=true_false_thresh,
            color="orange",
            linestyle=":",
            label=f"True to false threshold ({true_false_thresh})",
        )

        # Add more x-axis ticks for better precision
        plt.xticks(np.linspace(min(x_values), max(x_values), num=10, endpoint=True))

        # Labels
        plt.ylabel(foot_names[i])
        plt.grid(True, alpha=0.3)

        # Only show legend on the first subplot
        if i == 0:
            plt.legend(loc="upper right")

        # Only show x label on the last subplot
        if i == contact.shape[1] - 1:
            plt.xlabel(x_label)

    plt.suptitle(
        "Feet Contact Probabilities\n(Requires 3 consecutive frames above 'False to true' or below 'True to false' thresholds to change state)",
        fontsize=16,
    )
    plt.tight_layout()

    # Save the combined figure
    output_path = os.path.join(
        output_dir, f"all_contact_probabilities_{trial_name}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved combined contact plot to {output_path}")

    plt.close()

    return {
        "contact": contact,
        "filtered_contact": filtered_contact,
        "debounced_contact": debounced_contacts.numpy(),
        "frame_ids": frame_ids,
        "x_values": x_values,
        "foot_names": foot_names,
    }


def debounced_threshold(
    v_mask,
    false_true_thresh_heel=0.5,
    true_false_thresh_heel=0.5,
    false_true_thresh_big_toe=0.5,
    true_false_thresh_big_toe=0.5,
    min_stretch_len=1,
):
    """
    Apply a debounced threshold to a TxN matrix to reduce noise in contact detection.

    This implements a proper hysteresis state machine that considers the direction
    of movement to avoid rapid toggling between contact states.

    Args:
        v_mask: A TxN matrix of contact probabilities
        false_true_thresh_heel: Threshold for switching from False to True (no contact -> contact)
        true_false_thresh_heel: Threshold for switching from True to False (contact -> no contact)
        false_true_thresh_big_toe: Threshold for switching from False to True (no contact -> contact)
        true_false_thresh_big_toe: Threshold for switching from True to False (contact -> no contact)
        min_stretch_len: Minimum length of a stretch to trigger a state change

    Returns:
        Debounced binary contact mask (torch.bool)

    Hysteresis Logic:
        - When state is False: only switch to True if value goes ABOVE false_true_thresh
        - When state is True: only switch to False if value goes BELOW true_false_thresh
        - Values between thresholds maintain current state (no switching)
    """
    T, N = v_mask.shape
    debounced = torch.zeros_like(v_mask, dtype=torch.bool)

    foot_names = ["Left Big Toe", "Left Heel", "Right Big Toe", "Right Heel"]

    for n in range(N):

        if "Toe" in foot_names[n]:
            false_true_thresh = false_true_thresh_big_toe
            true_false_thresh = true_false_thresh_big_toe
        else:
            false_true_thresh = false_true_thresh_heel
            true_false_thresh = true_false_thresh_heel

        column = v_mask[:, n]
        # Initialize state based on first value
        # Use clear hysteresis logic even for initialization
        first_val = column[0]
        if first_val > false_true_thresh:
            current_state = True
        else:
            current_state = False

        # Counters for debouncing
        consecutive_frames_wanting_change = 0
        target_state = None

        for t in range(T):
            current_value = column[t]

            # check is the line of probability is increasing or decreasing over the last 5 frames
            if t > 1:
                minus = t - 5 if t - 5 > 0 else t - (t - 1)
                last_5_values = column[minus:t]
                # check if the last 5 values are increasing or decreasing
                # print("t: ", t)
                # print("current_value: ", current_value)
                # print("last_5_values: ", last_5_values)
                if torch.all(last_5_values < current_value):
                    is_increasing = True
                else:
                    is_increasing = False

            # Determine what state this value "wants" based on hysteresis
            if current_state == False:
                # When False, only consider switching to True if above false_true_thresh
                if current_value > false_true_thresh and is_increasing:
                    desired_state = True
                else:
                    desired_state = False  # Stay False
            else:  # current_state == True
                # When True, only consider switching to False if below true_false_thresh
                if current_value < true_false_thresh and not is_increasing:
                    desired_state = False
                else:
                    desired_state = True  # Stay True

            # Handle debouncing
            if desired_state != current_state:
                # We want to change state
                if target_state == desired_state:
                    # We're continuing to want the same state change
                    consecutive_frames_wanting_change += 1
                else:
                    # We want a different state change than before
                    target_state = desired_state
                    consecutive_frames_wanting_change = 1

                # Check if we've wanted this change long enough
                if consecutive_frames_wanting_change >= min_stretch_len:
                    current_state = desired_state
                    consecutive_frames_wanting_change = 0
                    target_state = None
            else:
                # We don't want to change state, reset counters
                consecutive_frames_wanting_change = 0
                target_state = None

            debounced[t, n] = current_state

    return debounced


def process_all_videos(
    base_dir,
    output_dir=None,
    false_true_thresh=0.5,
    true_false_thresh=0.5,
    use_seconds=True,
):
    """
    Process all videos in the base directory and generate contact probability plots.

    Args:
        base_dir (str): Base directory containing all video data directories
        output_dir (str): Directory to save all output plots
        false_true_thresh (float): High threshold for switching to contact
        true_false_thresh (float): Low threshold for switching from contact
        use_seconds (bool): If True, show time in seconds, otherwise show frame numbers
    """
    if output_dir is None:
        output_dir = os.path.join(base_dir, "contact_plots")
        os.makedirs(output_dir, exist_ok=True)

    # Find all directories containing wham_output.pkl
    pkl_files = []
    for root, dirs, files in os.walk(base_dir):
        if "wham_output.pkl" in files:
            pkl_files.append(os.path.join(root, "wham_output.pkl"))

    logger.info(f"Found {len(pkl_files)} WHAM output files")

    # Process each PKL file
    for pkl_file in pkl_files:
        try:
            # Create a specific output directory for this video
            video_name = os.path.basename(os.path.dirname(pkl_file))
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            logger.info(f"Processing {pkl_file}")
            visualize_contact_probabilities(
                pkl_path=pkl_file,
                output_dir=video_output_dir,
                false_true_thresh=false_true_thresh,
                true_false_thresh=true_false_thresh,
                use_seconds=use_seconds,
            )
        except Exception as e:
            logger.error(f"Error processing {pkl_file}: {e}")


def create_contact_debug_video(
    pkl_path=None,
    data_dir=None,
    video_path=None,
    output_path=None,
    false_true_thresh_heel=0.5,
    true_false_thresh_heel=0.5,
    false_true_thresh_big_toe=0.5,
    true_false_thresh_big_toe=0.5,
    use_seconds=True,
    min_stretch_len=1,
    use_raw_probabilities=False,
):
    """
    Create a debug video with contact information overlaid on the original video.

    Args:
        pkl_path (str): Direct path to the WHAM output.pkl file
        data_dir (str): Directory containing the WHAM output.pkl file
        video_path (str): Path to the original video file
        output_path (str): Path for the output debug video
        false_true_thresh (float): Threshold for switching from non-contact to contact
        true_false_thresh (float): Threshold for switching from contact to non-contact
        use_seconds (bool): If True, show time in seconds, otherwise show frame numbers
        min_stretch_len (int): Minimum stretch length for debounced threshold (default: 1)
        use_raw_probabilities (bool): If True, use raw probabilities without filtering or debouncing
    """
    # Handle direct PKL path or data directory
    if pkl_path is not None:
        wham_regression_results_pth = pkl_path
        data_dir = os.path.dirname(pkl_path)
    elif data_dir is not None:
        wham_regression_results_pth = os.path.join(data_dir, "wham_output.pkl")
    else:
        logger.error("Either pkl_path or data_dir must be provided")
        return

    # Load WHAM results
    try:
        wham_regression_results = ut.load_results(wham_regression_results_pth)
        logger.info(f"Loaded WHAM results from {wham_regression_results_pth}")
    except FileNotFoundError:
        logger.error(f"Could not find WHAM results at {wham_regression_results_pth}")
        return
    except Exception as e:
        logger.error(f"Error loading WHAM results: {e}")
        return

    # Find video file if not provided
    if video_path is None:
        video_files = [
            f
            for f in os.listdir(data_dir)
            if f.endswith((".mp4", ".avi", ".mov")) and "sync" not in f
        ]
        if not video_files:
            logger.error(f"No suitable video files found in {data_dir}")
            return
        video_path = os.path.join(data_dir, video_files[0])

    # Set default output path
    if output_path is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(data_dir, f"{video_name}_contact_debug.mp4")

    # Load video and get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

    # Process WHAM results
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

    # Get contact information
    contact = wham_result["contact"]

    # Get frame IDs
    if "frame_ids" in wham_result:
        frame_ids = wham_result["frame_ids"]
    elif "frame_id" in wham_result:
        frame_ids = wham_result["frame_id"]
    else:
        frame_ids = np.arange(len(contact))

    # Apply filtering and thresholding only if not using raw probabilities
    if use_raw_probabilities:
        filtered_contact = contact  # Use raw contact as filtered for display purposes
        debounced_contacts = None  # Not used when showing raw probabilities
        logger.info("Using raw probabilities without filtering or debouncing")
    else:
        filtered_contact = ut.filter_array(
            contact, order=4, cutoff_freq=6, sampling_rate=fps
        )
        contact_tensor = torch.from_numpy(filtered_contact)
        debounced_contacts = debounced_threshold(
            contact_tensor,
            false_true_thresh_heel=false_true_thresh_heel,
            true_false_thresh_heel=true_false_thresh_heel,
            false_true_thresh_big_toe=false_true_thresh_big_toe,
            true_false_thresh_big_toe=true_false_thresh_big_toe,
            min_stretch_len=min_stretch_len,
        )

    # Contact point names and colors
    foot_names = ["Left Big Toe", "Left Heel", "Right Big Toe", "Right Heel"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR format

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    contact_frame_idx = 0

    logger.info(f"Creating debug video with {len(frame_ids)} contact frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create overlay background for text
        overlay = frame.copy()

        # Add semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Add frame number and time
        if use_seconds:
            time_text = f"Time: {frame_number / fps:.2f}s"
        else:
            time_text = f"Frame: {frame_number}"

        cv2.putText(
            frame,
            time_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Check if current frame has contact data
        if (
            contact_frame_idx < len(frame_ids)
            and frame_number == frame_ids[contact_frame_idx]
        ):
            # Display contact information for each foot
            for i, (foot_name, color) in enumerate(zip(foot_names, colors)):
                y_pos = 70 + i * 30

                # Get contact values
                raw_contact = contact[contact_frame_idx, i]

                if use_raw_probabilities:
                    # Display only raw probability
                    prob_text = f"Probability: {raw_contact:.3f}"
                    cv2.putText(
                        frame,
                        f"{foot_name}:",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        prob_text,
                        (200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                else:
                    # Display filtered and binary contact information
                    filtered_contact_val = filtered_contact[contact_frame_idx, i]
                    binary_contact = debounced_contacts[contact_frame_idx, i].item()

                    # Create status text
                    status = "CONTACT" if binary_contact else "NO CONTACT"
                    status_color = (
                        (0, 255, 0) if binary_contact else (0, 0, 255)
                    )  # Green for contact, red for no contact

                    # Display foot name and status
                    cv2.putText(
                        frame,
                        f"{foot_name}:",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        status,
                        (200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        status_color,
                        2,
                    )

                    # Display probability values
                    prob_text = (
                        f"Raw: {raw_contact:.2f} Filt: {filtered_contact_val:.2f}"
                    )
                    cv2.putText(
                        frame,
                        prob_text,
                        (300, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

            # Add threshold information only if not using raw probabilities
            if not use_raw_probabilities:
                cv2.putText(
                    frame,
                    f"Thresholds heel: F->T: {false_true_thresh_heel:.2f}, T->F: {true_false_thresh_heel:.2f}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    f"Thresholds big toe: F->T: {false_true_thresh_big_toe:.2f}, T->F: {true_false_thresh_big_toe:.2f}",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            else:
                # Show that we're using raw probabilities
                cv2.putText(
                    frame,
                    "Raw probabilities (no filtering/debouncing)",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            contact_frame_idx += 1
        else:
            # No contact data for this frame
            cv2.putText(
                frame,
                "No contact data",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (128, 128, 128),
                2,
            )

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    logger.info(f"Debug video saved to: {output_path}")
    return output_path


def create_feet_debug_video(
    folder_path,
    false_true_thresh_heel=0.5,
    true_false_thresh_heel=0.5,
    false_true_thresh_big_toe=0.5,
    true_false_thresh_big_toe=0.5,
    min_stretch_len=1,
    use_raw_probabilities=False,
):
    """
    Create a debug video with contact information from a folder containing output.mp4 and wham_output.pkl.
    The output video will always be named feet.mp4.

    Args:
        folder_path (str): Path to folder containing output.mp4 and wham_output.pkl
        false_true_thresh (float): Threshold for switching from non-contact to contact
        true_false_thresh (float): Threshold for switching from contact to non-contact
        min_stretch_len (int): Minimum stretch length for debounced threshold (default: 1)
        use_raw_probabilities (bool): If True, use raw probabilities without filtering or debouncing

    Returns:
        str: Path to the created feet.mp4 file, or None if failed
    """
    folder_path = Path(folder_path)

    # Check if folder exists
    if not folder_path.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return None

    # Look for required files
    video_path = folder_path / "output.mp4"
    pkl_path = folder_path / "wham_output.pkl"
    output_path = folder_path / "feet.mp4"

    # Check if required files exist
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return None

    if not pkl_path.exists():
        logger.error(f"PKL file not found: {pkl_path}")
        return None

    logger.info(f"Found video: {video_path}")
    logger.info(f"Found PKL: {pkl_path}")
    logger.info(f"Output will be: {output_path}")

    # Create the debug video using existing function
    result = create_contact_debug_video(
        pkl_path=str(pkl_path),
        video_path=str(video_path),
        output_path=str(output_path),
        false_true_thresh_heel=false_true_thresh_heel,
        true_false_thresh_heel=true_false_thresh_heel,
        false_true_thresh_big_toe=false_true_thresh_big_toe,
        true_false_thresh_big_toe=true_false_thresh_big_toe,
        use_seconds=True,
        min_stretch_len=min_stretch_len,
        use_raw_probabilities=use_raw_probabilities,
    )

    if result:
        logger.info(f"Successfully created feet.mp4 at: {output_path}")
        return str(output_path)
    else:
        logger.error("Failed to create debug video")
        return None


if __name__ == "__main__":
    # python visualize_contact_probabilities.py --folder output/subject3/Session1/Cam3/walking1/walking1_trimmed
    parser = argparse.ArgumentParser(description="Visualize feet contact probabilities")
    parser.add_argument("--data_dir", type=str, help="Directory containing WHAM output")
    parser.add_argument(
        "--wham_pkl", type=str, help="Direct path to wham_output.pkl file"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to the original video file for debug video creation",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing output.mp4 and wham_output.pkl (will create feet.mp4)",
    )
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    parser.add_argument(
        "--output_video",
        type=str,
        help="Path for the output debug video (e.g., output.mp4)",
    )
    parser.add_argument(
        "--false_true_thresh_heel",
        type=float,
        default=0.38,
        help="Threshold for switching from no contact to contact",
    )
    parser.add_argument(
        "--true_false_thresh_heel",
        type=float,
        default=0.65,
        help="Threshold for switching from contact to no contact",
    )
    parser.add_argument(
        "--false_true_thresh_big_toe",
        type=float,
        default=0.5,
        help="Threshold for switching from no contact to contact",
    )
    parser.add_argument(
        "--true_false_thresh_big_toe",
        type=float,
        default=0.45,
        help="Threshold for switching from contact to no contact",
    )
    parser.add_argument(
        "--process_all",
        action="store_true",
        help="Process all videos in base directory",
    )
    parser.add_argument(
        "--create_debug_video",
        action="store_true",
        help="Create debug video with contact information overlaid",
    )
    parser.add_argument(
        "--use_seconds",
        action="store_true",
        default=True,
        help="Use time in seconds for x-axis (default)",
    )
    parser.add_argument(
        "--use_frames",
        action="store_true",
        help="Use frame numbers for x-axis instead of seconds",
    )
    parser.add_argument(
        "--min_stretch_len",
        type=int,
        default=1,
        help="Minimum stretch length for debounced threshold",
    )
    parser.add_argument(
        "--use_raw_probabilities",
        action="store_true",
        help="Use raw probabilities without filtering or debouncing",
    )

    args = parser.parse_args()

    # Determine whether to use seconds or frame numbers
    use_seconds = args.use_seconds and not args.use_frames

    if args.folder:
        # Use the new simplified folder-based approach
        output_video_path = create_feet_debug_video(
            folder_path=args.folder,
            false_true_thresh_heel=args.false_true_thresh_heel,
            true_false_thresh_heel=args.true_false_thresh_heel,
            false_true_thresh_big_toe=args.false_true_thresh_big_toe,
            true_false_thresh_big_toe=args.true_false_thresh_big_toe,
            min_stretch_len=args.min_stretch_len,
            use_raw_probabilities=args.use_raw_probabilities,
        )

        if output_video_path:
            print(f"feet.mp4 created successfully: {output_video_path}")
        else:
            print("Failed to create feet.mp4")
            exit(1)

    elif args.create_debug_video:
        # Create debug video with contact information
        if not args.data_dir and not args.wham_pkl:
            print(
                "Please provide either --data_dir or --wham_pkl for debug video creation"
            )
            exit(1)

        output_video_path = create_contact_debug_video(
            pkl_path=args.wham_pkl,
            data_dir=args.data_dir,
            video_path=args.video_path,
            output_path=args.output_video,
            false_true_thresh_heel=args.false_true_thresh_heel,
            true_false_thresh_heel=args.true_false_thresh_heel,
            false_true_thresh_big_toe=args.false_true_thresh_big_toe,
            true_false_thresh_big_toe=args.true_false_thresh_big_toe,
            use_seconds=use_seconds,
            use_raw_probabilities=args.use_raw_probabilities,
        )

        if output_video_path:
            print(f"Debug video created successfully: {output_video_path}")
        else:
            print("Failed to create debug video")

    elif args.process_all:
        if not args.data_dir:
            print("Please provide --data_dir when using --process_all")
            exit(1)
        process_all_videos(
            base_dir=args.data_dir,
            output_dir=args.output_dir,
            false_true_thresh=args.false_true_thresh_heel,  # Use heel threshold as default
            true_false_thresh=args.true_false_thresh_heel,  # Use heel threshold as default
            use_seconds=use_seconds,
        )
    else:
        if not args.data_dir and not args.wham_pkl:
            print("Please provide either --data_dir or --wham_pkl")
            exit(1)
        visualize_contact_probabilities(
            pkl_path=args.wham_pkl,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            false_true_thresh=args.false_true_thresh_heel,  # Use heel threshold as default
            true_false_thresh=args.true_false_thresh_heel,  # Use heel threshold as default
            use_seconds=use_seconds,
        )
