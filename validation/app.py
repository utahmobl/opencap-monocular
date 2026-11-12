import os
import sys
import webbrowser
import streamlit as st
from pathlib import Path
import time
from loguru import logger
from time_sync import run_time_sync
from space_sync import run_space_sync_and_ik
from ik_analysis import run_ik_analysis
from marker_analysis import run_marker_analysis
import pandas as pd
import torch
import gc
import json
from visualization.utils import generateVisualizerJson
from visualization.automation import automate_recording
import yaml
from hashlib import md5
import signal
import psutil
import threading
import random
import glob
import numpy as np


def aggregate_case_results(case_dir):
    """
    Aggregate all translation_error.csv files from a case directory into a single DataFrame.
    
    This function finds all individual translation_error.csv files (one per video) in the case
    directory structure and combines them into a single aggregated result.
    
    Parameters:
        case_dir: Path to the case directory (e.g., output/case_002_walking)
        
    Returns:
        dict: Dictionary containing:
            - 'df': Aggregated DataFrame with all results
            - 'avg_mae_mm': Mean MAE in mm across all videos
            - 'avg_mae_degrees': Mean MAE in degrees across all videos
            - 'marker_mae': Mean marker MAE if available
            - 'foot_marker_error': Mean foot marker error if available
            - 'num_videos': Number of videos found
            - 'files': List of CSV files found
        Returns None if no valid results found.
    """
    # Find all translation_error.csv files in the case directory
    all_csvs = glob.glob(os.path.join(case_dir, '**/translation_error.csv'), recursive=True)
    
    if not all_csvs:
        logger.warning(f"No translation_error.csv files found in {case_dir}")
        return None
    
    # Read and concatenate all CSVs
    dfs = []
    valid_files = []
    for csv_file in all_csvs:
        try:
            df = pd.read_csv(csv_file)
            if 'global_mae_mm' in df.columns and 'global_mae_degrees' in df.columns:
                dfs.append(df)
                valid_files.append(csv_file)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")
            continue
    
    if not dfs:
        logger.warning(f"No valid translation_error.csv files with required columns found in {case_dir}")
        return None
    
    # Concatenate all DataFrames
    aggregated_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate aggregated metrics
    avg_mae_mm = aggregated_df['global_mae_mm'].mean()
    avg_mae_degrees = aggregated_df['global_mae_degrees'].mean()
    
    # Calculate marker MAE if available
    marker_mae = 0
    if 'marker_mae_mm' in aggregated_df.columns:
        marker_mae = aggregated_df['marker_mae_mm'].mean()
    
    # Calculate foot marker error (ankles + toes) if available
    foot_marker_error = 0
    if 'ankles' in aggregated_df.columns and 'toes' in aggregated_df.columns:
        foot_marker_error = (aggregated_df['ankles'].mean() + aggregated_df['toes'].mean()) / 2
    elif 'marker_mae_mm' in aggregated_df.columns:
        # Fallback to overall marker error if individual foot markers not available
        foot_marker_error = aggregated_df['marker_mae_mm'].mean()
    
    logger.info(f"Aggregated {len(valid_files)} videos from {case_dir}: "
                f"MAE mm: {avg_mae_mm:.2f}, MAE degrees: {avg_mae_degrees:.2f}")
    
    return {
        'df': aggregated_df,
        'avg_mae_mm': avg_mae_mm,
        'avg_mae_degrees': avg_mae_degrees,
        'marker_mae': marker_mae,
        'foot_marker_error': foot_marker_error,
        'num_videos': len(valid_files),
        'files': valid_files
    }


def format_exception_details(e, context=""):
    """
    Format exception details with full traceback for better debugging.
    
    Parameters:
        e: The exception object
        context: Optional context string describing where the error occurred
        
    Returns:
        dict: Dictionary with error details including traceback
    """
    import traceback
    import sys
    
    # Get the full traceback
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    full_traceback = ''.join(tb_lines)
    
    # Get the specific line where error occurred
    tb = traceback.extract_tb(exc_traceback)
    if tb:
        last_frame = tb[-1]
        error_location = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        error_line = last_frame.line if last_frame.line else "Unknown line"
    else:
        error_location = "Unknown location"
        error_line = "Unknown line"
    
    error_details = {
        "type": type(e).__name__,
        "message": str(e),
        "location": error_location,
        "line": error_line,
        "traceback": full_traceback,
        "context": context
    }
    
    return error_details


def display_error_in_streamlit(error_details, case_num=None):
    """
    Display formatted error information in Streamlit.
    
    Parameters:
        error_details: Dictionary from format_exception_details()
        case_num: Optional case number for context
    """
    case_info = f" in case {case_num}" if case_num else ""
    
    st.error(f"❌ Error{case_info}: {error_details['type']}: {error_details['message']}")
    
    if error_details.get('context'):
        st.error(f"🔍 Context: {error_details['context']}")
    
    st.error(f"📍 Location: {error_details['location']}")
    st.error(f"📝 Line: {error_details['line']}")
    
    # Show full traceback in expander
    with st.expander("🔍 Show Full Traceback"):
        st.code(error_details['traceback'], language="python")


def log_error_details(error_details, logger_instance, case_num=None):
    """
    Log formatted error information.
    
    Parameters:
        error_details: Dictionary from format_exception_details()
        logger_instance: Logger instance to use
        case_num: Optional case number for context
    """
    case_info = f" in case {case_num}" if case_num else ""
    
    logger_instance.error(f"Error{case_info}")
    logger_instance.error(f"Error type: {error_details['type']}")
    logger_instance.error(f"Error message: {error_details['message']}")
    
    if error_details.get('context'):
        logger_instance.error(f"Context: {error_details['context']}")
    
    logger_instance.error(f"Error location: {error_details['location']}")
    logger_instance.error(f"Error line: {error_details['line']}")
    logger_instance.error(f"Full traceback:\n{error_details['traceback']}")


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection with better error handling"""
    try:
        if torch.cuda.is_available():
            # Get initial memory stats
            allocated_before = torch.cuda.memory_allocated() / 1024**2
            cached_before = torch.cuda.memory_reserved() / 1024**2

            # Clear cache multiple times
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Force garbage collection
            gc.collect()

            # Final clear
            torch.cuda.empty_cache()

            # Log memory improvement
            allocated_after = torch.cuda.memory_allocated() / 1024**2
            cached_after = torch.cuda.memory_reserved() / 1024**2

            logger.info(
                f"GPU memory cleared: {allocated_before:.1f}MB -> {allocated_after:.1f}MB allocated, "
                f"{cached_before:.1f}MB -> {cached_after:.1f}MB cached"
            )
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {e}")

    # System memory cleanup
    try:
        gc.collect()
        # Force Python to release memory back to system
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception as e:
        logger.debug(f"Could not trim system memory: {e}")


# Add signal handler for graceful shutdown (only if in main thread)
def signal_handler(signum, frame):
    logger.info("Received shutdown signal, cleaning up...")
    clear_gpu_memory()
    gc.collect()
    sys.exit(0)


# Only set up signal handlers if we're in the main thread and not in Streamlit
try:
    # Check if we're running in Streamlit (Streamlit sets this environment variable)
    if (
        "STREAMLIT_SERVER_PORT" not in os.environ
        and threading.current_thread() is threading.main_thread()
    ):
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("Signal handlers set up successfully")
    else:
        logger.info("Running in Streamlit environment, skipping signal handler setup")
except (ValueError, OSError) as e:
    logger.warning(f"Could not set up signal handlers: {e}")


# Memory monitoring function
def check_memory_usage():
    """Monitor memory usage and clean up if necessary"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        if memory_mb > 12000:  # If using more than 12GB
            logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
            if "memory_warnings" in st.session_state:
                st.session_state.memory_warnings += 1
            clear_gpu_memory()
            gc.collect()
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking memory: {e}")
        return False


# Add repo path to system path for imports
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_path)

# Import NAS sync function
try:
    from nas.sync_utils import sync as sync_to_nas

    nas_sync_available = True
except ImportError:
    logger.warning("NAS sync module not available. Files will be saved locally only.")
    nas_sync_available = False

# streamlit run  validation/app.py

global filter_freq, weights_opt2

# Add session state for memory tracking
if "memory_warnings" not in st.session_state:
    st.session_state.memory_warnings = 0

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimization import run_optimization
from WHAM.demo import main_wham
from utils.utilsCameraPy3 import getVideoRotation
from utils.convert_to_avi import convert_to_avi

# Flag to control NAS usage
use_nas = True
run_wham = True


def get_or_create_case(weights_opt2, filter_freq, movement, current_params=None):
    """
    Get existing case ID or create a new one for the given parameters.

    Parameters:
        weights_opt2 (dict): Dictionary of weight parameters
        filter_freq (dict): Dictionary of filter frequencies
        movement (str): Movement type (walking, squats, STS)
        current_params (dict, optional): Override parameters for HP search

    Returns:
        str: Case identifier (e.g., "case_001_walking")
    """
    # Determine which parameters to use (handle HP search case)
    if current_params is not None:
        weights = current_params["weights"]
    else:
        weights = weights_opt2[movement]

    # Get current filter frequency for this movement
    freq = filter_freq[movement]

    # Create parameter dictionary to hash
    params_dict = {"weights": weights, "filter_freq": freq, "movement": movement}

    # Convert params to a sorted string for consistent hashing
    param_str = json.dumps(params_dict, sort_keys=True)

    # Create a hash of the parameters for quick lookup
    param_hash = md5(param_str.encode()).hexdigest()[:8]

    # Path to cases file for this movement
    cases_file = os.path.join(repo_path, "output", f"cases_{movement}.json")

    # Check if cases file exists and load it
    if os.path.exists(cases_file):
        with open(cases_file, "r") as f:
            cases = json.load(f)
    else:
        cases = {}

    # Check if this parameter combination already exists
    for case_id, case_params in cases.items():
        if "param_hash" in case_params and case_params["param_hash"] == param_hash:
            case_dir = os.path.join(repo_path, "output", case_id)
            return case_id, case_dir

    # If not found, create a new case ID
    new_case_num = len(cases) + 1
    new_case_id = f"case_{new_case_num:03d}_{movement}"

    # Add the new case to the cases dictionary
    cases[new_case_id] = {
        "weights": weights,
        "filter_freq": freq,
        "movement": movement,
        "param_hash": param_hash,
    }

    # Save the updated cases file
    with open(cases_file, "w") as f:
        json.dump(cases, f, indent=4)

    # Create the case directory if it doesn't exist
    case_dir = os.path.join(repo_path, "output", new_case_id)
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    # Also save parameters directly in the case directory
    params_file = os.path.join(case_dir, "parameters.json")
    with open(params_file, "w") as f:
        json.dump(params_dict, f, indent=4)

    return new_case_id, case_dir


def get_output_path(subject, session, cam, video):
    """Get the output path, always using local path for file operations"""
    # Always return the local path for file operations
    local_path = os.path.join(repo_path, "output", subject, session, cam, video)

    # Create the directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)

    return local_path


def sync_to_nas_if_enabled(msg="Syncing to NAS..."):
    """Sync to NAS if enabled"""
    if use_nas and nas_sync_available:
        logger.info(msg)
        success = sync_to_nas()
        if success:
            logger.info("Sync completed successfully")
        else:
            logger.warning("Sync to NAS failed")
        return success
    return False


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        st.write("Current GPU memory usage:")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        st.write(f"Allocated: {allocated:.2f} MB")
        st.write(f"Reserved: {reserved:.2f} MB")


def display_memory_status():
    """Display current memory status in Streamlit sidebar"""
    with st.sidebar:
        st.subheader("🔧 System Status")

        # System memory
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_color = (
                "🟢" if memory_mb < 8000 else "🟡" if memory_mb < 12000 else "🔴"
            )
            st.metric("System Memory", f"{memory_mb:.0f} MB", delta=None)
            st.write(
                f"{memory_color} Status: {'Good' if memory_mb < 8000 else 'Warning' if memory_mb < 12000 else 'Critical'}"
            )
        except Exception as e:
            st.write(f"❌ Memory check failed: {e}")

        # GPU memory
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                gpu_color = (
                    "🟢" if allocated < 2000 else "🟡" if allocated < 4000 else "🔴"
                )
                st.metric("GPU Memory", f"{allocated:.0f} MB", delta=None)
                st.write(
                    f"{gpu_color} GPU Status: {'Good' if allocated < 2000 else 'Warning' if allocated < 4000 else 'Critical'}"
                )
            except Exception as e:
                st.write(f"❌ GPU check failed: {e}")
        else:
            st.write("❌ CUDA not available")

        # Memory warnings counter
        if st.session_state.memory_warnings > 0:
            st.warning(f"⚠️ Memory warnings: {st.session_state.memory_warnings}")

        # Clear memory button
        if st.button("🧹 Clear Memory"):
            clear_gpu_memory()
            st.success("Memory cleared!")


def load_combinations_from_yaml(yaml_path, selected_movements):
    """
    Load parameter combinations from parameters_search.yaml file.
    
    This function handles the YAML file which has duplicate keys by parsing it as text.
    
    Parameters:
        yaml_path (str): Path to the parameters_search.yaml file
        selected_movements (list): List of movements to load combinations for
        
    Returns:
        dict: Dictionary of combinations by movement
    """
    import re
    
    if not os.path.exists(yaml_path):
        logger.error(f"File not found: {yaml_path}")
        return {}
    
    try:
        with open(yaml_path, "r") as f:
            content = f.read()
        
        combinations = {}
        
        # Map movement names to parameter keys in the YAML
        movement_to_param_key = {
            "walking": "walking_parameters",
            "squats": "squat_parameters",
            "STS": "sts_parameters"
        }
        
        for movement in selected_movements:
            param_key = movement_to_param_key.get(movement)
            
            if param_key is None:
                logger.warning(f"No parameter key mapping for movement: {movement}")
                continue
            
            # Find the section for this movement
            section_start = content.find(f"{param_key}:")
            if section_start == -1:
                logger.warning(f"Section {param_key} not found in {yaml_path}")
                continue
            
            # Find the next section or end of file
            next_section_patterns = ["walking_parameters:", "squat_parameters:", "sts_parameters:", "other_parameters:"]
            section_end = len(content)
            for pattern in next_section_patterns:
                pos = content.find(pattern, section_start + 1)
                if pos != -1 and pos < section_end:
                    section_end = pos
            
            section_content = content[section_start:section_end]
            
            # Determine the weight key name
            weight_key = f"weights_opt2_{movement}" if movement != "STS" else "weights_opt2_sts"
            
            # Parse each weight configuration in this section
            # Pattern matches: weights_opt2_X: followed by indented key: value pairs
            # Use [ \t]+ to match spaces or tabs for indentation
            weight_blocks = re.findall(
                rf'{weight_key}:\s*\n((?:[ \t]+\w+:[ \t]*\d+[ \t]*\n)+)',
                section_content,
                re.MULTILINE
            )
            
            movement_combinations = []
            
            for block in weight_blocks:
                # Parse the weight values
                weights_dict = {}
                for line in block.strip().split('\n'):
                    line = line.strip()
                    if ':' in line and not line.startswith('#'):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            weights_dict[key] = int(value)
                        except ValueError:
                            logger.warning(f"Could not parse value for {key}: {value}")
                            continue
                
                if weights_dict:
                    movement_combinations.append({
                        "movement": movement,
                        "weights": weights_dict
                    })
            
            combinations[movement] = movement_combinations
            logger.info(f"Loaded {len(movement_combinations)} combinations for {movement}")
        
        return combinations
        
    except Exception as e:
        logger.error(f"Error loading combinations from {yaml_path}: {e}", exc_info=True)
        return {}


def generate_parameter_combinations(weight_ranges, max_combinations):
    """Generate parameter combinations for each movement independently"""
    # Define the fixed values to test
    test_values = [1, 10, 100]

    # Generate combinations per movement
    combinations_per_movement = {}

    for movement in weight_ranges:
        # Generate all possible combinations for this movement
        movement_combinations = []
        weight_keys = list(weight_ranges[movement].keys())

        # Generate all combinations using the fixed test values
        import itertools

        value_combinations = list(
            itertools.product(test_values, repeat=len(weight_keys))
        )

        # Convert combinations to dictionary format
        for values in value_combinations:
            weights_dict = {key: value for key, value in zip(weight_keys, values)}
            movement_combinations.append(
                {"movement": movement, "weights": weights_dict}
            )

        combinations_per_movement[movement] = movement_combinations

    return combinations_per_movement


def analyze_hp_results(movement):
    """Analyze hyperparameter search results for a given movement"""
    cases_file = os.path.join(repo_path, "output", f"cases_{movement}.json")
    if not os.path.exists(cases_file):
        return None

    # Load cases file to get parameters for each case
    with open(cases_file, "r") as f:
        cases = json.load(f)

    results_dict = {}

    # Process each case
    for case_num, params in cases.items():
        results_path = os.path.join(repo_path, "output", case_num, "results.csv")
        if not os.path.exists(results_path):
            continue

        # Load results CSV
        df = pd.read_csv(results_path)

        # Calculate average errors
        avg_mae_mm = df["global_mae_mm"].mean()
        avg_mae_degrees = df["global_mae_degrees"].mean()
        
        # Calculate foot marker error (ankles + toes)
        foot_marker_error = 0
        if "ankles" in df.columns and "toes" in df.columns:
            foot_marker_error = (df["ankles"].mean() + df["toes"].mean()) / 2
        elif "marker_mae_mm" in df.columns:
            # Fallback to overall marker error if individual foot markers not available
            foot_marker_error = df["marker_mae_mm"].mean()
        
        total_error = avg_mae_mm + avg_mae_degrees + (foot_marker_error / 5)

        # Store results with parameters
        results_dict[case_num] = {
            "parameters": params,
            "avg_mae_mm": avg_mae_mm,
            "avg_mae_degrees": avg_mae_degrees,
            "foot_marker_error": foot_marker_error,
            "total_error": total_error,
        }

    # Sort results by total error
    sorted_results = dict(
        sorted(results_dict.items(), key=lambda x: x[1]["total_error"])
    )

    # Save sorted results to JSON
    results_file = os.path.join(repo_path, "output", f"hp_results_{movement}.json")
    with open(results_file, "w") as f:
        json.dump(sorted_results, f, indent=4)

    return sorted_results


def display_best_combinations():
    """Display the top 3 combinations for each movement"""
    st.subheader("Best Hyperparameter Combinations")

    for movement in ["walking", "squats", "STS"]:
        st.write(f"\n### {movement.capitalize()}")

        results = analyze_hp_results(movement)
        if results is None:
            st.write(f"No results found for {movement}")
            continue

        # Get top 3 combinations
        top_3 = dict(list(results.items())[:3])

        for rank, (case_num, data) in enumerate(top_3.items(), 1):
            st.write(f"\n**Rank {rank}** (Case: {case_num})")

            # Create a formatted display of the results
            col1, col2 = st.columns(2)
            with col1:
                st.write("Parameters:")
                st.json(data["parameters"])

            with col2:
                st.write("Results:")
                metrics = {
                    "Average MAE (mm)": f"{data['avg_mae_mm']:.2f}",
                    "Average MAE (degrees)": f"{data['avg_mae_degrees']:.2f}",
                    "Total Error": f"{data['total_error']:.2f}",
                }
                for metric, value in metrics.items():
                    st.write(f"{metric}: {value}")

            st.write("---")


# @st.cache_resource(ttl=3600)  # Cache for 1 hour
def run(
    movement=None,
    test_subjects=None,
    test_cameras=None,
    estimate_local_only=True,
    rerun=False,
    force_rerun_analysis=False,
    run_2cams_analysis=True,
    success_message="Mono pipeline completed successfully!",
    output_case_path=None,
    current_params=None,
    specific_videos=None,  # New parameter: list of video info dicts to process
):
    # movement is 'walking', 'squat', 'sts'
    # test_subjects is a list of subject IDs to test on, e.g. ['subject1', 'subject2']
    # trial_path is the specific case folder path for HP search
    
    # NOTE: try-except blocks have been commented out for debugging
    # This allows the full traceback to be shown to identify the exact error location

    smoothness_diff_n = 1

    # Clear GPU memory at the start
    clear_gpu_memory()

    # Add memory monitoring
    check_memory_usage()

    results = {}
    start_time = time.time()

    output_csv_path = None

    # try:  # COMMENTED OUT FOR DEBUGGING - to see exact error location
    if True:  # Replace try with if True to maintain indentation
        ############## RUN MONO ##############

        if selected_subject == "ALL":
            if test_subjects is not None:
                subjects = test_subjects
            else:
                subjects = [
                    d
                    for d in os.listdir(validation_videos_path)
                    if os.path.isdir(os.path.join(validation_videos_path, d))
                ]
            print("Subjects:", subjects)
        else:
            # print("NOT ALL - video")
            subjects = [selected_subject]

        for subject_idx, subject in enumerate(subjects):
            # Monitor memory usage between subjects
            if check_memory_usage():
                st.warning(f"High memory usage detected before processing {subject}")

            subject_path = os.path.join(validation_videos_path, subject)
            if selected_session == "ALL":
                sessions = [
                    d
                    for d in os.listdir(os.path.join(subject_path, "VideoData"))
                    if os.path.isdir(os.path.join(subject_path, "VideoData", d))
                ]
                print("Sessions:", sessions)
            else:
                # print("NOT ALL - camera")
                sessions = [selected_session]

            for session_idx, session in enumerate(sessions):
                session_path = os.path.join(subject_path, "VideoData", session)
                if selected_cam == "ALL":
                    if test_cameras is not None:
                        cams = test_cameras
                    elif only_cam1_cam3:
                        cams = ["Cam1", "Cam3"]
                    # elif only_cam1:
                    #     cams = ['Cam1']
                    elif only_cam3:
                        cams = ["Cam3"]
                    else:
                        cams = [
                            d
                            for d in os.listdir(session_path)
                            if os.path.isdir(os.path.join(session_path, d))
                        ]
                else:
                    cams = [selected_cam]

                for cam_idx, cam in enumerate(cams):
                    # Monitor memory before each camera
                    if check_memory_usage():
                        st.warning(
                            f"High memory usage detected before processing {cam}"
                        )

                    cam_path = os.path.join(session_path, cam)
                    if selected_video == "ALL":
                        videos = [
                            d
                            for d in os.listdir(cam_path)
                            if os.path.isdir(os.path.join(cam_path, d))
                        ]
                        videos = [
                            v
                            for v in videos
                            if "extrinsics" not in v
                            and "_syncdWithMocap" not in v
                            and "static" not in v
                            and "DJ" not in v
                        ]

                        trimmed_videos = [v for v in videos if "trimmed" in v]
                        if trimmed_videos != []:
                            videos = trimmed_videos

                        print("Videos:", videos)
                    else:
                        # print("NOT ALL - video")
                        videos = [selected_video]

                    for video_idx, video in enumerate(videos):
                        # Monitor memory before each video
                        memory_high = check_memory_usage()
                        if memory_high:
                            st.warning(
                                f"High memory usage detected before processing {video}"
                            )

                        video_path = os.path.join(cam_path, video)
                        if selected_video_name == "ALL":
                            video_names = [
                                f
                                for f in os.listdir(video_path)
                                if f.endswith(".avi")
                                and not "extrinsics" in f
                                and not "static" in f
                                and not "_syncdWithMocap" in f
                                and "DJ" not in f
                            ]
                            trimmed_videos = [v for v in video_names if "trimmed" in v]
                            if trimmed_videos != []:
                                video_names = trimmed_videos
                            print("Video names:", video_names)
                        else:
                            # print("NOT ALL - video name")
                            video_names = [selected_video_name]

                        if movement is not None:
                            video_names = [
                                v for v in video_names if movement.lower() in v.lower()
                            ]
                            logger.info(
                                f"Filtered video names for movement {movement}: {video_names}"
                            )

                        for video_name in video_names:
                            video_path = os.path.join(video_path, video_name)
                            # check if the video exists
                            if not os.path.exists(video_path):
                                logger.error(f"Video path does not exist: {video_path}")
                                continue

                            # Skip if specific_videos is provided and this video is not in the list
                            if specific_videos is not None:
                                video_found = False
                                for target_video in specific_videos:
                                    if (target_video["subject"] == subject and 
                                        target_video["session"] == session and 
                                        target_video["cam"] == cam and 
                                        target_video["video"] == video and 
                                        target_video["video_name"] == video_name):
                                        video_found = True
                                        break
                                
                                if not video_found:
                                    continue

                            st.write(f"{subject} - {session} - {cam} - {video_name}")

                            # Modify the output CSV path logic
                            if output_case_path:
                                output_csv = os.path.join(
                                    output_case_path, "results.csv"
                                )
                            else:  # Normal run mode
                                output_csv = os.path.join(
                                    repo_path,
                                    "output",
                                    subject,
                                    session,
                                    cam,
                                    video,
                                    "results.csv",
                                )

                            # check if the csv exists
                            if os.path.exists(output_csv):
                                csv_path = output_csv
                                # open the csv file and check if the current video is already in the file
                                df = pd.read_csv(csv_path)

                                # remove all rows with NaN values

                                df = df.dropna()
                                df.to_csv(output_csv, index=False)

                                if (
                                    len(
                                        df.loc[
                                            (df["subject"] == subject)
                                            & (df["session"] == session)
                                            & (df["cam"] == cam)
                                            & (df["movement"] == video)
                                        ]
                                    )
                                    > 0
                                ):
                                    if (
                                        not df.loc[
                                            (df["subject"] == subject)
                                            & (df["session"] == session)
                                            & (df["cam"] == cam)
                                            & (df["movement"] == video),
                                            "global_mae_mm",
                                        ]
                                        .isnull()
                                        .values.any()
                                    ):
                                        st.write(
                                            "Video already in the csv file. Skipping..."
                                        )
                                        # write the path of the csv file to the console
                                        st.write(f"CSV path: {output_csv}")
                                        st.write("----------------")
                                        continue
                                    else:
                                        st.write(
                                            "Video already in the csv file but with NaN values. Running again..."
                                        )
                                        # # remove the row
                                        # df = df.loc[~((df["subject"] == subject) & (df["session"] == session) & (df["cam"] == cam) & (df["movement"] == video))]
                                        # df.to_csv(output_csv, index=False)
                            else:
                                csv_path = None

                            case_path = os.path.join(
                                output_case_path, subject, session, cam, video
                            )
                            if not os.path.exists(case_path):
                                os.makedirs(case_path)

                            # Check if the results already exist
                            files_exist = os.path.exists(
                                os.path.join(
                                    case_path,
                                    "OpenSim",
                                    "IK",
                                    "shiftedIK",
                                    "translation_error.txt",
                                )
                            )

                            # If files exist and we're not forcing rerun, skip entirely
                            if files_exist and not force_rerun_analysis:
                                st.write("Files already in the case path. Skipping...")
                                st.write("----------------")
                                continue

                            # If files exist and we are forcing rerun, skip to analysis only
                            if files_exist and force_rerun_analysis:
                                st.write(
                                    "Files exist, but force rerun analysis is enabled. Running only analysis steps..."
                                )

                                # Find required paths for analysis
                                video_is_trimmed = False
                                if "trimmed" in video_name:
                                    video_is_trimmed = True

                                # Get path to the IK output motion file
                                ik_folder = os.path.join(
                                    case_path, "OpenSim", "IK", "shiftedIK"
                                )
                                motion_files = [
                                    f
                                    for f in os.listdir(ik_folder)
                                    if f.endswith("_sync.mot")
                                ]
                                if not motion_files:
                                    st.error(
                                        f"No IK motion file found in {ik_folder}. Cannot run analysis."
                                    )
                                    continue

                                pathOutputMotion = os.path.join(
                                    ik_folder, motion_files[0]
                                )

                                # Run IK analysis
                                st.write("Running IK analysis...")
                                ik_results = run_ik_analysis(
                                    subject,
                                    session,
                                    cam,
                                    video,
                                    trimmed=video_is_trimmed,
                                    output_case_path=case_path,
                                    run_wham=True,
                                    run_2cams=run_2cams_analysis,
                                )

                                # Unpack results depending on whether WHAM and 2CAMS are included
                                if len(ik_results) == 8:  # Both WHAM and 2CAMS
                                    (
                                        ik_results_degrees,
                                        ik_results_mm,
                                        ik_results_degrees_wham,
                                        ik_results_mm_wham,
                                        ik_results_degrees_2cams,
                                        ik_results_mm_2cams,
                                        results_path,
                                        output_csv_path,
                                    ) = ik_results
                                    # Print all results
                                    st.write("IK Analysis Results:")
                                    st.write(
                                        f"Mono - MAE degrees: {ik_results_degrees}, MAE mm: {ik_results_mm}"
                                    )
                                    st.write(
                                        f"WHAM - MAE degrees: {ik_results_degrees_wham}, MAE mm: {ik_results_mm_wham}"
                                    )
                                    st.write(
                                        f"2CAMS - MAE degrees: {ik_results_degrees_2cams}, MAE mm: {ik_results_mm_2cams}"
                                    )
                                elif len(ik_results) == 6:  # Either WHAM or 2CAMS
                                    if run_wham:
                                        (
                                            ik_results_degrees,
                                            ik_results_mm,
                                            ik_results_degrees_wham,
                                            ik_results_mm_wham,
                                            results_path,
                                            output_csv_path,
                                        ) = ik_results
                                        st.write("IK Analysis Results:")
                                        st.write(
                                            f"Mono - MAE degrees: {ik_results_degrees}, MAE mm: {ik_results_mm}"
                                        )
                                        st.write(
                                            f"WHAM - MAE degrees: {ik_results_degrees_wham}, MAE mm: {ik_results_mm_wham}"
                                        )
                                    else:  # Must be 2CAMS
                                        (
                                            ik_results_degrees,
                                            ik_results_mm,
                                            ik_results_degrees_2cams,
                                            ik_results_mm_2cams,
                                            results_path,
                                            output_csv_path,
                                        ) = ik_results
                                        st.write("IK Analysis Results:")
                                        st.write(
                                            f"Mono - MAE degrees: {ik_results_degrees}, MAE mm: {ik_results_mm}"
                                        )
                                        st.write(
                                            f"2CAMS - MAE degrees: {ik_results_degrees_2cams}, MAE mm: {ik_results_mm_2cams}"
                                        )
                                else:  # Just Mono
                                    (
                                        ik_results_degrees,
                                        ik_results_mm,
                                        results_path,
                                        output_csv_path,
                                    ) = ik_results
                                    st.write(
                                        "IK Analysis Results: MAE degrees:",
                                        ik_results_degrees,
                                        " MAE mm:",
                                        ik_results_mm,
                                    )

                                logger.info("IK analysis done")

                                st.write("Analysis rerun completed.")
                                st.write("----------------")
                                continue

                            # Continue with the normal pipeline if files don't exist
                            # Log the weights being used for this trial
                            st.write("\nUsing weights for this trial:")
                            if movement in weights_opt2:
                                st.write(f"Movement: {movement}")
                                
                                # Use current parameters for HP search if provided
                                if current_params is not None:
                                    current_weights = current_params["weights"]
                                else:
                                    current_weights = weights_opt2[movement]
                                
                                weight_cols = st.columns(len(current_weights))
                                for i, (key, value) in enumerate(
                                    current_weights.items()
                                ):
                                    with weight_cols[i]:
                                        st.metric(label=key, value=value)
                                st.write(f"Filter frequency: {filter_freq[movement]}")

                            local_output_path = os.path.join(
                                repo_path, "output", subject, session, cam, video
                            )

                            # Convert MOV to AVI if necessary
                            if video_path.lower().endswith(".mov"):
                                logger.info(f"Converting MOV file to AVI: {video_path}")
                                video_path = convert_to_avi(video_path)
                                logger.info(f"Conversion complete. New video path: {video_path}")

                            inputs_wham = {
                                "calib_path": "examples/walking4/calib.txt",
                                "video_path": video_path,
                                "output_path": local_output_path,
                                "visualize": False,
                                "estimate_local_only": estimate_local_only,
                                "save_pkl": True,
                                "run_smplify": True,
                                "rerun": rerun,
                            }

                            results = main_wham(**inputs_wham)

                            logger.info("Wham done")
                            logger.info(
                                f"Time taken for Wham: {time.time() - start_time:.2f} seconds"
                            )

                            st.write("Wham done")

                            # Clear memory after WHAM processing
                            clear_gpu_memory()
                            check_memory_usage()

                            metadata = os.path.join(
                                subject_path, "sessionMetadata.yaml"
                            )
                            with open(metadata, "rb") as f:
                                metadata = yaml.load(f, Loader=yaml.FullLoader)

                            height_m = metadata["height_m"] 
                            mass_kg = metadata["mass_kg"]
                            sex = metadata["sex"]
                            logger.info(
                                f"Height: {height_m} m, Mass: {mass_kg} kg, Sex: {sex}"
                            )
                            video_name_trimmed = video_name.split(".avi")[0]
                            if "_trimmed" in video_name:
                                video_name_trimmed = video_name.split(".avi")[0]

                            results_path = os.path.join(
                                repo_path,
                                "output",
                                subject,
                                session,
                                cam,
                                video,
                                video_name_trimmed,
                            )

                            step_start_time = time.time()

                            torch.cuda.empty_cache()

                            # Determine video rotation
                            rotation = getVideoRotation(video_path)
                            logger.info(f"Rotation: {rotation}")

                            inputs_optimization = {
                                "data_dir": results_path,
                                "trial_name": video,
                                "height_m": height_m,
                                "mass_kg": mass_kg,
                                "sex": sex,
                                "intrinsics_pth": "examples/Intrinsics/iphone12Pro_intrinsics.pickle",
                                "case": "5",
                                "trc_rot": {"x": 0, "y": 46.5, "z": 0},
                                "run_opensim_original_wham": True,
                                "run_opensim_opt2": True,
                                "use_gpu": True,
                                "filter_freq": filter_freq[movement],
                                "static_cam": False,
                                "n_iter_opt2": 75,
                                "print_loss_terms": False,
                                "smoothness_diff_n": smoothness_diff_n,
                                "plotting": True,
                                "output_path": case_path,
                                "video_path": video_path,
                                "activity": movement,
                                "weights_opt2": current_weights,  # Use current weights for HP search
                                "rotation": rotation,
                            }

                            results_optimization = run_optimization(
                                **inputs_optimization
                            )
                            logger.info("Optimization done")
                            logger.info(
                                f"Time taken for Optimization: {time.time() - start_time:.2f} seconds"
                            )

                            if results_optimization['trc_file'] is None:
                                st.warning("Optimization failed. Skipping to next trial...")
                                continue
                                
                            st.write("Optimization done")

                            lag, max_corr, graph_path = run_time_sync(
                                subject,
                                session,
                                cam,
                                movement=video,
                                output_case_path=case_path,
                                visualize=False,
                            )

                            logger.info(f"Lag: {lag}, Max Correlation: {max_corr}")

                            if single_trial:
                                st.write("Lag:", lag)
                                st.write("Max Correlation:", max_corr)
                                # st.write("Graph Path:", graph_path)
                                # open the html file in a new tab
                                webbrowser.open_new_tab(graph_path)

                            marker_wham_path = None

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
                                    marker_wham_path_sub_folder = os.path.join(
                                        marker_wham_path_folder, folder
                                    )
                                    break

                            for file in os.listdir(marker_wham_path_sub_folder):
                                if "wham_result.trc" in file:
                                    marker_wham_path = os.path.join(
                                        marker_wham_path_sub_folder, file
                                    )
                                    break

                            logger.info(f"Marker wham path: {marker_wham_path}")

                            movement_path, synced_path, pathOutputMotion = (
                                run_space_sync_and_ik(
                                    subject,
                                    session,
                                    cam,
                                    movement=video,
                                    marker_wham_path=marker_wham_path,
                                    output_case_path=case_path,
                                )
                            )
                            
                            # Check if space sync and IK failed
                            if movement_path is None or synced_path is None or pathOutputMotion is None:
                                logger.error("Space sync and IK failed - skipping this video")
                                st.error("Space sync and IK failed - skipping this video")
                                continue
                            
                            logger.info("Space sync and IK analysis done")

                            st.write("Space sync and IK analysis done")

                            # if single_trial:
                            #     st.write("Output Motion Path:", pathOutputMotion)
                            #     st.write("Synced Path:", synced_path)
                            #     st.write("Movement Path:", movement_path)

                            video_is_trimmed = False
                            if "trimmed" in video_name:
                                video_is_trimmed = True

                            # Update ik_analysis call to include run_2cams parameter and handle all return values
                            ik_results = run_ik_analysis(
                                subject,
                                session,
                                cam,
                                video,
                                trimmed=video_is_trimmed,
                                output_case_path=case_path,
                                run_wham=True,
                                run_2cams=run_2cams_analysis,
                            )

                            # Unpack results depending on whether WHAM and 2CAMS are included
                            if len(ik_results) == 8:  # Both WHAM and 2CAMS
                                (
                                    ik_results_degrees,
                                    ik_results_mm,
                                    ik_results_degrees_wham,
                                    ik_results_mm_wham,
                                    ik_results_degrees_2cams,
                                    ik_results_mm_2cams,
                                    results_path,
                                    output_csv_path,
                                ) = ik_results
                                # Print all results
                                st.write("IK Analysis Results:")
                                st.write(
                                    f"Mono - MAE degrees: {ik_results_degrees}, MAE mm: {ik_results_mm}"
                                )
                                st.write(
                                    f"WHAM - MAE degrees: {ik_results_degrees_wham}, MAE mm: {ik_results_mm_wham}"
                                )
                                st.write(
                                    f"2CAMS - MAE degrees: {ik_results_degrees_2cams}, MAE mm: {ik_results_mm_2cams}"
                                )
                            elif len(ik_results) == 6:  # Either WHAM or 2CAMS
                                if "run_wham" in locals() and run_wham:
                                    (
                                        ik_results_degrees,
                                        ik_results_mm,
                                        ik_results_degrees_wham,
                                        ik_results_mm_wham,
                                        results_path,
                                        output_csv_path,
                                    ) = ik_results
                                    st.write("IK Analysis Results:")
                                    st.write(
                                        f"Mono - MAE degrees: {ik_results_degrees}, MAE mm: {ik_results_mm}"
                                    )
                                    st.write(
                                        f"WHAM - MAE degrees: {ik_results_degrees_wham}, MAE mm: {ik_results_mm_wham}"
                                    )
                                else:  # Must be 2CAMS
                                    (
                                        ik_results_degrees,
                                        ik_results_mm,
                                        ik_results_degrees_2cams,
                                        ik_results_mm_2cams,
                                        results_path,
                                        output_csv_path,
                                    ) = ik_results
                                    st.write("IK Analysis Results:")
                                    st.write(
                                        f"Mono - MAE degrees: {ik_results_degrees}, MAE mm: {ik_results_mm}"
                                    )
                                    st.write(
                                        f"2CAMS - MAE degrees: {ik_results_degrees_2cams}, MAE mm: {ik_results_mm_2cams}"
                                    )
                            else:  # Just Mono
                                (
                                    ik_results_degrees,
                                    ik_results_mm,
                                    results_path,
                                    output_csv_path,
                                ) = ik_results
                                st.write(
                                    "IK Analysis Results: MAE degrees:",
                                    ik_results_degrees,
                                    " MAE mm:",
                                    ik_results_mm,
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

                            # Display marker analysis results
                            if marker_results is not None:
                                logger.info("Marker analysis done")
                                st.write("Marker Analysis Results:")
                                st.write(
                                    f"Mono - Marker MAE: {marker_results['mono']} mm"
                                )
                                if "wham" in marker_results:
                                    st.write(
                                        f"WHAM - Marker MAE: {marker_results['wham']} mm"
                                    )
                                if "2cams" in marker_results:
                                    st.write(
                                        f"2CAMS - Marker MAE: {marker_results['2cams']} mm"
                                    )
                            else:
                                logger.warning("Marker analysis failed")
                                st.warning("Marker analysis could not be completed")

                            # visualize the results (skip during hyperparameter search)
                            if current_params is None:  # Only generate visualizations for normal runs, not HP search
                                # Mocap
                                mocap_folder_path = os.path.join(movement_path, "mocap")
                                # find a .osim in this folder
                                mocap_model_file = None
                                for file in os.listdir(mocap_folder_path):
                                    if file.endswith(".osim"):
                                        mocap_model_file = os.path.join(
                                            mocap_folder_path, file
                                        )
                                        break
                                mocap_if_file = None
                                for file in os.listdir(mocap_folder_path):
                                    if file.endswith(".mot"):
                                        mocap_if_file = os.path.join(
                                            mocap_folder_path, file
                                        )
                                        break
                                assert (
                                    mocap_model_file is not None
                                ), "No .osim file found in the mocap folder"
                                assert (
                                    mocap_if_file is not None
                                ), "No .mot file found in the mocap folder"

                                output_mocap_json_path = os.path.join(
                                    mocap_folder_path, "mocap.json"
                                )

                                try:
                                    generateVisualizerJson(
                                        modelPath=mocap_model_file,
                                        ikPath=mocap_if_file,
                                        jsonOutputPath=output_mocap_json_path,
                                        vertical_offset=0,
                                    )
                                    assert os.path.exists(
                                        output_mocap_json_path
                                    ), "Mocap json file not created"
                                except Exception as e:
                                    st.warning(f"Error generating mocap json file: {e}")
                                    st.warning(f"Mocap json file not created")
                                    st.warning(f"Mocap json file not created")

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
                                try:
                                    generateVisualizerJson(
                                        modelPath=model_mono_file,
                                        ikPath=pathOutputMotion,
                                        jsonOutputPath=output_mono_json_path,
                                        vertical_offset=0,
                                    )
                                    assert os.path.exists(
                                        output_mono_json_path
                                    ), "Mono json file not created"
                                except Exception as e:
                                    st.warning(f"Error generating mono json file: {e}")
                                    st.warning(f"Mono json file not created")
                                    st.warning(f"Mono json file not created")

                                if marker_wham_path is not None:
                                    output_wham_json_path = os.path.join(
                                        "/", *pathOutputMotion.split("/")[:-1], "wham.json"
                                    )
                                    model_wham_file = model_mono_file
                                    wham_file = pathOutputMotion.replace(
                                        ".mot", "_wham.mot"
                                    )
                                    assert (
                                        wham_file is not None
                                    ), "No wham_result.mot file found in the wham_result folder"

                                    generateVisualizerJson(
                                        modelPath=model_wham_file,
                                        ikPath=wham_file,
                                        jsonOutputPath=output_wham_json_path,
                                        vertical_offset=0,
                                    )

                                include_2cams_viz = True
                                if include_2cams_viz:
                                    # 2cams
                                    output_2cams_json_path = os.path.join(
                                        "/", *pathOutputMotion.split("/")[:-1], "2cams.json"
                                    )

                                    # The 2-camera IK file should correspond to the base video name (trial name),
                                    # without the suffixes added by the mono pipeline (like _5 for case, or _sync for time synchronization).
                                    file_original_name = f"{video}.mot"

                                    twocams_file = os.path.join(
                                        f"LabValidation_withVideos1/{subject}/OpenSimData/Video/HRNet/2-cameras/IK/{file_original_name}"
                                    )
                                    model_2cams_file = os.path.join(
                                        f"LabValidation_withVideos1/{subject}/OpenSimData/Video/HRNet/2-cameras/Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim"
                                    )

                                    if not os.path.exists(twocams_file):
                                        st.warning(
                                            f"2-camera IK file not found, skipping visualization: {twocams_file}"
                                        )
                                    else:
                                        try:
                                            generateVisualizerJson(
                                                modelPath=model_2cams_file,
                                                ikPath=twocams_file,
                                                jsonOutputPath=output_2cams_json_path,
                                                vertical_offset=0,
                                            )
                                            assert os.path.exists(
                                                output_2cams_json_path
                                            ), "2cams json file not created"
                                        except Exception as e:
                                            st.warning(
                                                f"Error generating 2cams json file: {e}"
                                            )
                                            st.warning(f"2cams json file not created")
                                            st.warning(f"2cams json file not created")

                                logger.info("Json files created")
                            else:
                                logger.info("Skipping visualization generation for hyperparameter search")

                            # Skip video generation during hyperparameter search
                            if current_params is None:  # Only generate videos for normal runs, not HP search
                                output_video_path_mono = os.path.join(
                                    movement_path, "viewer_mono.webm"
                                )

                                json_files = [output_mocap_json_path, output_mono_json_path]

                                if marker_wham_path is not None:
                                    json_files.append(output_wham_json_path)

                                if include_2cams_viz:
                                    json_files.append(output_2cams_json_path)

                                viz = False
                                if viz:
                                    automate_recording(
                                        json_files,
                                        output_video_path=output_video_path_mono,
                                        num_loops=1,
                                    )
                                    st.write("Visualization created")
                            else:
                                logger.info("Skipping video generation for hyperparameter search")

                            # TODO integrate the GRF run and analysis here as well by using the api function or add the code here

                            st.markdown("----------------")

                            # Clear memory after each video processing
                            clear_gpu_memory()
                            check_memory_usage()

                            # Log completion
                            logger.info(
                                f"Completed processing video {video_idx + 1}/{len(videos)} for {subject}-{session}-{cam}"
                            )

    # except RuntimeError as e:  # COMMENTED OUT FOR DEBUGGING
    #     if "CUDA out of memory" in str(e):
    #         st.error(
    #             "GPU ran out of memory. Try processing fewer videos at once or use a machine with more GPU memory."
    #         )
    #         logger.error(f"CUDA OOM error: {str(e)}")
    #     raise e
    # except Exception as e:
    #     logger.error(f"Unexpected error in processing: {str(e)}")
    #     st.error(f"An error occurred: {str(e)}")
    #     raise e
    # finally:  # COMMENTED OUT FOR DEBUGGING
    if True:  # Replace finally with if True to maintain indentation
        # Always clear GPU memory when done
        clear_gpu_memory()

        # Final memory check and cleanup
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Final memory usage: {memory_mb:.1f} MB")

            # Force final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_error:
            logger.error(f"Error in final cleanup: {cleanup_error}")

    logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")
    st.write("Results csv path:", output_csv_path)

    if output_csv_path is not None:
        df = pd.read_csv(output_csv_path)

        # Define a function to highlight values above the threshold
        def highlight_above_threshold(s, thresholds):
            return [
                "color: red" if col in thresholds and s[col] > thresholds[col] else ""
                for col in s.index
            ]

        # Apply the style to the DataFrame
        styled_df = df.style.apply(
            highlight_above_threshold, thresholds=thresholds, axis=1
        )

        # Filter the DataFrame to keep only rows and columns with red values
        def filter_above_threshold(df, thresholds):
            # Keep the first 5 columns as identifiers
            identifier_columns = df.columns[:5]
            # Find columns with values above the threshold
            columns_above_threshold = [
                col
                for col in df.columns
                if col in thresholds and (df[col] > thresholds[col]).any()
            ]
            # Combine identifier columns with columns above the threshold
            columns_to_keep = list(identifier_columns) + columns_above_threshold
            # Filter rows with any value above the threshold
            rows_above_threshold = df[columns_to_keep].apply(
                lambda row: any(
                    (row[5:] > thresholds[col]).any()
                    for col in row.index[5:]
                    if col in thresholds
                ),
                axis=1,
            )
            return df.loc[rows_above_threshold, columns_to_keep]

        filtered_df = filter_above_threshold(df, thresholds)

        # Apply the style to the filtered DataFrame
        styled_filtered_df = filtered_df.style.apply(
            highlight_above_threshold, thresholds=thresholds, axis=1
        )

        # Display the styled filtered DataFrame
        st.dataframe(styled_filtered_df)

    st.success(success_message)

    return output_csv_path


def check_existing_results(case_dir, subject, session, cam, video, video_name):
    """
    Check if results already exist for a specific video in a case directory.
    
    Returns:
        dict: Status information about existing results
    """
    case_path = os.path.join(case_dir, subject, session, cam, video)
    
    # Check if case directory exists
    if not os.path.exists(case_path):
        return {
            "exists": False,
            "complete": False,
            "partial": False,
            "missing_videos": [],
            "existing_videos": []
        }
    
    # Check for the key result file that indicates complete processing
    translation_error_file = os.path.join(
        case_path,
        "OpenSim",
        "IK",
        "shiftedIK",
        "translation_error.txt",
    )
    
    # Check for CSV results file
    csv_file = os.path.join(case_path, "results.csv")
    
    if os.path.exists(translation_error_file) and os.path.exists(csv_file):
        # Results exist, check if this specific video is in the CSV
        try:
            df = pd.read_csv(csv_file)
            df = df.dropna()  # Remove rows with NaN values
            
            # Check if this specific video exists in results
            video_exists = len(
                df.loc[
                    (df["subject"] == subject)
                    & (df["session"] == session)
                    & (df["cam"] == cam)
                    & (df["movement"] == video)
                ]
            ) > 0
            
            if video_exists:
                # Check if the results are complete (no NaN values)
                video_results = df.loc[
                    (df["subject"] == subject)
                    & (df["session"] == session)
                    & (df["cam"] == cam)
                    & (df["movement"] == video)
                ]
                
                has_complete_results = not video_results["global_mae_mm"].isnull().values.any()
                
                if has_complete_results:
                    return {
                        "exists": True,
                        "complete": True,
                        "partial": False,
                        "missing_videos": [],
                        "existing_videos": [video_name]
                    }
                else:
                    return {
                        "exists": True,
                        "complete": False,
                        "partial": True,
                        "missing_videos": [video_name],
                        "existing_videos": []
                    }
            else:
                return {
                    "exists": True,
                    "complete": False,
                    "partial": True,
                    "missing_videos": [video_name],
                    "existing_videos": []
                }
                
        except Exception as e:
            logger.warning(f"Error reading CSV file {csv_file}: {e}")
            return {
                "exists": True,
                "complete": False,
                "partial": True,
                "missing_videos": [video_name],
                "existing_videos": []
            }
    
    elif os.path.exists(translation_error_file):
        # Files exist but no CSV, consider it partial
        return {
            "exists": True,
            "complete": False,
            "partial": True,
            "missing_videos": [video_name],
            "existing_videos": []
        }
    
    else:
        # No results exist
        return {
            "exists": False,
            "complete": False,
            "partial": False,
            "missing_videos": [video_name],
            "existing_videos": []
        }


def get_videos_for_movement(validation_videos_path, subject, session, cam, movement):
    """
    Get all videos for a specific movement from the validation dataset.
    
    Returns:
        list: List of video names that match the movement
    """
    videos = []
    
    if subject == "ALL":
        subjects = [
            d for d in os.listdir(validation_videos_path)
            if os.path.isdir(os.path.join(validation_videos_path, d))
        ]
    else:
        subjects = [subject]
    
    for subj in subjects:
        subject_path = os.path.join(validation_videos_path, subj)
        if not os.path.exists(subject_path):
            continue
            
        if session == "ALL":
            sessions = [
                d for d in os.listdir(os.path.join(subject_path, "VideoData"))
                if os.path.isdir(os.path.join(subject_path, "VideoData", d))
            ]
        else:
            sessions = [session]
        
        for sess in sessions:
            session_path = os.path.join(subject_path, "VideoData", sess)
            if not os.path.exists(session_path):
                continue
                
            if cam == "ALL":
                cams = [
                    d for d in os.listdir(session_path)
                    if os.path.isdir(os.path.join(session_path, d))
                ]
            else:
                cams = [cam]
            
            for camera in cams:
                cam_path = os.path.join(session_path, camera)
                if not os.path.exists(cam_path):
                    continue
                
                # Get all video folders
                video_folders = [
                    d for d in os.listdir(cam_path)
                    if os.path.isdir(os.path.join(cam_path, d))
                    and "extrinsics" not in d
                    and "_syncdWithMocap" not in d
                    and "static" not in d
                    and "DJ" not in d
                ]
                
                # Prefer trimmed videos if available
                trimmed_videos = [v for v in video_folders if "trimmed" in v]
                if trimmed_videos:
                    video_folders = trimmed_videos
                
                for video_folder in video_folders:
                    video_path = os.path.join(cam_path, video_folder)
                    
                    # Get video files
                    video_files = [
                        f for f in os.listdir(video_path)
                        if f.endswith(".avi")
                        and not "extrinsics" in f
                        and not "static" in f
                        and not "_syncdWithMocap" in f
                        and "DJ" not in f
                    ]
                    
                    # Prefer trimmed videos if available
                    trimmed_video_files = [v for v in video_files if "trimmed" in v]
                    if trimmed_video_files:
                        video_files = trimmed_video_files
                    
                    # Filter by movement
                    for video_file in video_files:
                        if movement.lower() in video_file.lower():
                            videos.append({
                                "subject": subj,
                                "session": sess,
                                "cam": camera,
                                "video": video_folder,
                                "video_name": video_file
                            })
    
    return videos


def get_saved_combination_files():
    """
    Get list of saved combination files.
    
    Returns:
        list: List of file paths for saved combinations
    """
    output_dir = os.path.join(repo_path, "output", "hp_combinations")
    if not os.path.exists(output_dir):
        return []
    
    files = []
    for filename in os.listdir(output_dir):
        if filename.startswith("hp_combinations_") and filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            files.append(filepath)
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files


def load_combinations_from_file(filepath):
    """
    Load combinations from a saved JSON file.
    
    Parameters:
        filepath (str): Path to the saved combinations file
        
    Returns:
        tuple: (combinations, metadata) or (None, None) if failed
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        combinations = data.get("combinations", {})
        metadata = data.get("metadata", {})
        
        logger.info(f"Loaded combinations from {filepath}")
        logger.info(f"Metadata: {metadata}")
        
        return combinations, metadata
    except Exception as e:
        logger.error(f"Error loading combinations from {filepath}: {e}")
        return None, None


def save_combinations_to_file(combinations, movements, subject, session, max_combinations, random_seed):
    """
    Save generated combinations to a JSON file for later loading.
    
    Parameters:
        combinations (dict): Dictionary of combinations by movement
        movements (list): List of movements
        subject (str): Subject selection
        session (str): Session selection
        max_combinations (int): Maximum combinations per movement
        random_seed (int): Random seed used
    """
    # Create metadata for the combinations
    metadata = {
        "timestamp": time.time(),
        "movements": movements,
        "subject": subject,
        "session": session,
        "max_combinations": max_combinations,
        "random_seed": random_seed,
        "total_combinations": sum(len(combinations.get(movement, [])) for movement in movements)
    }
    
    # Create the data structure to save
    save_data = {
        "metadata": metadata,
        "combinations": combinations
    }
    
    # Create filename with timestamp and parameters
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    subject_str = subject.replace(" ", "_")
    session_str = session.replace(" ", "_")
    filename = f"hp_combinations_{timestamp_str}_{subject_str}_{session_str}.json"
    
    # Save to output directory
    output_dir = os.path.join(repo_path, "output", "hp_combinations")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=4, default=str)
    
    logger.info(f"Saved {len(combinations)} movement combinations to {filepath}")
    return filepath


def check_combinations_compatibility(combinations, selected_movements):
    """
    Check if loaded combinations are compatible with currently selected movements.
    
    Parameters:
        combinations (dict): Loaded combinations
        selected_movements (list): Currently selected movements
        
    Returns:
        tuple: (is_compatible, missing_movements, extra_movements)
    """
    loaded_movements = set(combinations.keys())
    selected_movements_set = set(selected_movements)
    
    missing_movements = selected_movements_set - loaded_movements
    extra_movements = loaded_movements - selected_movements_set
    
    is_compatible = len(missing_movements) == 0
    
    return is_compatible, list(missing_movements), list(extra_movements)


def save_combinations_with_status(combinations, movements, subject, session, max_combinations, random_seed, hp_results=None):
    """
    Save generated combinations to a JSON file with run status and scores for later loading.
    
    Parameters:
        combinations (dict): Dictionary of combinations by movement
        movements (list): List of movements
        subject (str): Subject selection
        session (str): Session selection
        max_combinations (int): Maximum combinations per movement
        random_seed (int): Random seed used
        hp_results (dict): Optional existing results to include run status and scores
    """
    # Create metadata for the combinations
    metadata = {
        "timestamp": time.time(),
        "movements": movements,
        "subject": subject,
        "session": session,
        "max_combinations": max_combinations,
        "random_seed": random_seed,
        "total_combinations": sum(len(combinations.get(movement, [])) for movement in movements)
    }
    
    # Enhance combinations with run status and scores
    enhanced_combinations = {}
    for movement, movement_combinations in combinations.items():
        enhanced_combinations[movement] = []
        
        for combo in movement_combinations:
            enhanced_combo = {
                "combination": combo,
                "ran": False,
                "score": None,
                "error": None,
                "status": "pending"
            }
            
            # Check if we have results for this combination
            if hp_results and movement in hp_results:
                for result in hp_results[movement]:
                    if (result.get("parameters", {}).get("weights") == combo["weights"] and
                        result.get("parameters", {}).get("movement") == combo["movement"]):
                        enhanced_combo["ran"] = True
                        enhanced_combo["score"] = result.get("total_error")
                        enhanced_combo["error"] = result.get("error")
                        enhanced_combo["status"] = result.get("status", "completed")
                        break
            
            enhanced_combinations[movement].append(enhanced_combo)
    
    # Create the data structure to save
    save_data = {
        "metadata": metadata,
        "combinations": enhanced_combinations
    }
    
    # Create filename with timestamp and parameters
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    subject_str = subject.replace(" ", "_")
    session_str = session.replace(" ", "_")
    filename = f"hp_combinations_with_status_{timestamp_str}_{subject_str}_{session_str}.json"
    
    # Save to output directory
    output_dir = os.path.join(repo_path, "output", "hp_combinations")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=4, default=str)
    
    logger.info(f"Saved {len(enhanced_combinations)} movement combinations with status to {filepath}")
    return filepath


def load_combinations_with_status(filepath):
    """
    Load combinations with run status and scores from a saved JSON file.
    
    Parameters:
        filepath (str): Path to the saved combinations file
        
    Returns:
        tuple: (combinations, metadata, run_status) or (None, None, None) if failed
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        enhanced_combinations = data.get("combinations", {})
        metadata = data.get("metadata", {})
        
        # Extract original combinations and run status
        combinations = {}
        run_status = {}
        
        for movement, enhanced_combos in enhanced_combinations.items():
            combinations[movement] = []
            run_status[movement] = []
            
            for enhanced_combo in enhanced_combos:
                combinations[movement].append(enhanced_combo["combination"])
                run_status[movement].append({
                    "ran": enhanced_combo["ran"],
                    "score": enhanced_combo["score"],
                    "error": enhanced_combo["error"],
                    "status": enhanced_combo["status"]
                })
        
        logger.info(f"Loaded combinations with status from {filepath}")
        logger.info(f"Metadata: {metadata}")
        
        return combinations, metadata, run_status
    except Exception as e:
        logger.error(f"Error loading combinations with status from {filepath}: {e}")
        return None, None, None


def update_combination_status(filepath, movement, combo_index, ran=True, score=None, error=None, status="completed"):
    """
    Update the run status and score for a specific combination in a saved file.
    
    Parameters:
        filepath (str): Path to the saved combinations file
        movement (str): Movement name
        combo_index (int): Index of the combination to update
        ran (bool): Whether the combination has been run
        score (float): Score/error value
        error (str): Error message if any
        status (str): Status of the run
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        enhanced_combinations = data.get("combinations", {})
        
        if movement in enhanced_combinations and combo_index < len(enhanced_combinations[movement]):
            enhanced_combinations[movement][combo_index]["ran"] = ran
            enhanced_combinations[movement][combo_index]["score"] = score
            enhanced_combinations[movement][combo_index]["error"] = error
            enhanced_combinations[movement][combo_index]["status"] = status
            
            # Update timestamp
            data["metadata"]["last_updated"] = time.time()
            
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4, default=str)
            
            logger.info(f"Updated combination {combo_index} for {movement} in {filepath}")
            return True
        else:
            logger.error(f"Invalid movement or combo_index: {movement}, {combo_index}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating combination status in {filepath}: {e}")
        return False


def get_unfinished_combinations(enhanced_combinations):
    """
    Get combinations that haven't been run yet.
    
    Parameters:
        enhanced_combinations (dict): Enhanced combinations with run status
        
    Returns:
        dict: Dictionary of unfinished combinations by movement
    """
    unfinished = {}
    
    for movement, enhanced_combos in enhanced_combinations.items():
        unfinished[movement] = []
        
        for enhanced_combo in enhanced_combos:
            if not enhanced_combo["ran"]:
                unfinished[movement].append(enhanced_combo["combination"])
    
    return unfinished


def get_combination_summary(enhanced_combinations):
    """
    Get a summary of combination run status.
    
    Parameters:
        enhanced_combinations (dict): Enhanced combinations with run status
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_combinations": 0,
        "ran_combinations": 0,
        "pending_combinations": 0,
        "error_combinations": 0,
        "best_scores": {},
        "movement_stats": {}
    }
    
    for movement, enhanced_combos in enhanced_combinations.items():
        movement_stats = {
            "total": len(enhanced_combos),
            "ran": 0,
            "pending": 0,
            "error": 0,
            "best_score": None
        }
        
        for enhanced_combo in enhanced_combos:
            summary["total_combinations"] += 1
            movement_stats["total"] += 1
            
            if enhanced_combo["ran"]:
                summary["ran_combinations"] += 1
                movement_stats["ran"] += 1
                
                if enhanced_combo["score"] is not None:
                    if movement_stats["best_score"] is None or enhanced_combo["score"] < movement_stats["best_score"]:
                        movement_stats["best_score"] = enhanced_combo["score"]
            else:
                summary["pending_combinations"] += 1
                movement_stats["pending"] += 1
            
            if enhanced_combo["status"] == "error":
                summary["error_combinations"] += 1
                movement_stats["error"] += 1
        
        summary["movement_stats"][movement] = movement_stats
        if movement_stats["best_score"] is not None:
            summary["best_scores"][movement] = movement_stats["best_score"]
    
    return summary


def get_saved_combination_files_with_status():
    """
    Get list of saved combination files with status.
    
    Returns:
        list: List of file paths for saved combinations with status
    """
    output_dir = os.path.join(repo_path, "output", "hp_combinations")
    if not os.path.exists(output_dir):
        return []
    
    files = []
    for filename in os.listdir(output_dir):
        if filename.startswith("hp_combinations_with_status_") and filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            files.append(filepath)
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files


def save_hp_search_progress(hp_results, combinations, metadata, progress_info):
    """
    Save hyperparameter search progress to allow resuming later.
    
    Parameters:
        hp_results (dict): Current results
        combinations (dict): Original combinations
        metadata (dict): Search metadata
        progress_info (dict): Progress tracking info
    """
    progress_data = {
        "timestamp": time.time(),
        "metadata": metadata,
        "combinations": combinations,
        "results": hp_results,
        "progress": progress_info
    }
    
    # Create filename
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    subject_str = metadata.get("subject", "Unknown").replace(" ", "_")
    session_str = metadata.get("session", "Unknown").replace(" ", "_")
    filename = f"hp_search_progress_{timestamp_str}_{subject_str}_{session_str}.json"
    
    # Save to output directory
    output_dir = os.path.join(repo_path, "output", "hp_search_progress")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(progress_data, f, indent=4, default=str)
    
    logger.info(f"Saved HP search progress to {filepath}")
    return filepath


def load_hp_search_progress(filepath):
    """
    Load hyperparameter search progress from file.
    
    Parameters:
        filepath (str): Path to progress file
        
    Returns:
        tuple: (hp_results, combinations, metadata, progress_info) or (None, None, None, None) if failed
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        hp_results = data.get("results", {})
        combinations = data.get("combinations", {})
        metadata = data.get("metadata", {})
        progress_info = data.get("progress", {})
        
        logger.info(f"Loaded HP search progress from {filepath}")
        return hp_results, combinations, metadata, progress_info
    except Exception as e:
        logger.error(f"Error loading HP search progress from {filepath}: {e}")
        return None, None, None, None


def get_hp_search_progress_files():
    """
    Get list of HP search progress files.
    
    Returns:
        list: List of file paths for progress files
    """
    output_dir = os.path.join(repo_path, "output", "hp_search_progress")
    if not os.path.exists(output_dir):
        return []
    
    files = []
    for filename in os.listdir(output_dir):
        if filename.startswith("hp_search_progress_") and filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            files.append(filepath)
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files


# Initialize global variables and paths
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
validation_videos_path = os.path.join(repo_path, "LabValidation_withVideos1")

output_nas_path = os.path.join(
    "smb-share:server=mobl-nas.mech.utah.edu,share=mobl-nas/", "Users/SGilon/Mono"
)

# Load parameters
params_path = os.path.join(repo_path, "params/parameters.yaml")
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

filter_freq = params["filter_freq"]
weights_opt2_walking = params["weights_opt2_walking"]
weights_opt2_squats = params["weights_opt2_squats"]
weights_opt2_sts = params["weights_opt2_sts"]

weights_opt2 = {
    "walking": weights_opt2_walking,
    "squats": weights_opt2_squats,
    "STS": weights_opt2_sts,
}

# Define global thresholds
thresholds = {
    "pelvis_tz_mm": 50,
    "pelvis_tx_mm": 50,
    "pelvis_ty_mm": 50,
    "global_mae_mm": 100,
    "global_mae_degrees": 10,
}

# Start UI code
st.title("Mono - Experiments Launcher")

# Display memory status in sidebar
display_memory_status()

# Create two main tabs
normal_tab, hp_search_tab = st.tabs(["Normal Launch", "Hyperparameter Search"])

with normal_tab:
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    validation_videos_path = os.path.join(repo_path, "LabValidation_withVideos1")

    # Change movement selector from selectbox to multiselect
    movements = ["walking", "squats", "STS"]
    selected_movements = st.multiselect(
        "Movements", movements, default=movements[:], key="normal_movement"
    )

    # Ensure at least one movement is selected
    if not selected_movements:
        st.warning("Please select at least one movement")
        st.stop()

    subjects = ["ALL"] + [
        d
        for d in os.listdir(validation_videos_path)
        if os.path.isdir(os.path.join(validation_videos_path, d))
    ]
    selected_subject = st.selectbox("Subject", subjects, key="normal_subject")

    if selected_subject != "ALL":
        subject_path = os.path.join(validation_videos_path, selected_subject)
        sessions = ["ALL"] + [
            d
            for d in os.listdir(os.path.join(subject_path, "VideoData"))
            if os.path.isdir(os.path.join(subject_path, "VideoData", d))
        ]
    else:
        sessions = ["ALL", "Session0", "Session1"]
    selected_session = st.selectbox("Session", sessions, key="normal_session")

    if selected_subject != "ALL" and selected_session != "ALL":
        session_path = os.path.join(subject_path, "VideoData", selected_session)
        cams = ["ALL"] + [
            d
            for d in os.listdir(session_path)
            if os.path.isdir(os.path.join(session_path, d))
        ]
    else:
        cams = ["ALL"]

    selected_cam = st.selectbox("Camera", cams, key="normal_cam")

    # add a checkbox to only show Cam1 and Cam3
    if selected_cam == "ALL":
        only_cam1_cam3 = st.checkbox("Only Cam1 and Cam3", value=False)
        # only_cam1 = st.checkbox("Only Cam1", value=True)
        only_cam3 = st.checkbox("Only Cam3", value=True)

    if (
        selected_subject != "ALL"
        and selected_session != "ALL"
        and selected_cam != "ALL"
    ):
        cam_path = os.path.join(session_path, selected_cam)
        videos = ["ALL"] + [
            d for d in os.listdir(cam_path) if os.path.isdir(os.path.join(cam_path, d))
        ]
    else:
        videos = ["ALL"]
    videos = [
        v
        for v in videos
        if "extrinsics" not in v and "static" not in v and "DJ" not in v
    ]

    # Update video filtering to check against any selected movement
    filtered_videos = [
        v
        for v in videos
        if v == "ALL"
        or any(movement.lower() in v.lower() for movement in selected_movements)
    ]
    selected_video = st.selectbox("Video", filtered_videos)

    if (
        selected_subject != "ALL"
        and selected_session != "ALL"
        and selected_cam != "ALL"
        and selected_video != "ALL"
    ):
        video_path = os.path.join(cam_path, selected_video)
        video_names = [f for f in os.listdir(video_path) if f.endswith(".avi")]
    else:
        video_names = ["ALL"]
    video_names = [
        v
        for v in video_names
        if "_syncdWithMocap" not in v and "static" not in v and "DJ" not in v
    ]

    # Update video names filtering to check against any selected movement
    filtered_video_names = [
        v
        for v in video_names
        if v == "ALL"
        or any(movement.lower() in v.lower() for movement in selected_movements)
    ]

    trimmed_videos = [v for v in filtered_video_names if "trimmed" in v]
    if trimmed_videos != []:
        filtered_video_names = trimmed_videos

    selected_video_name = st.selectbox("Video Name", filtered_video_names)

    single_trial = (
        True
        if selected_subject != "ALL"
        and selected_session != "ALL"
        and selected_cam != "ALL"
        and selected_video != "ALL"
        and selected_video_name != "ALL"
        else False
    )

    with st.expander("Parameters"):
        # Add WHAM settings
        st.subheader("WHAM Settings")
        wham_cols = st.columns(2)
        with wham_cols[0]:
            estimate_local_only = st.checkbox(
                "Estimate Local Only", value=False, key="normal_estimate_local"
            )
        with wham_cols[1]:
            rerun = st.checkbox("Rerun WHAM", value=False, key="normal_rerun")

        # Add analysis rerun settings
        st.subheader("Analysis Settings")
        analysis_cols = st.columns(2)
        with analysis_cols[0]:
            force_rerun_analysis = st.checkbox(
                "Force Rerun Analysis",
                value=False,
                help="Rerun IK and marker analysis even if files already exist",
            )
        with analysis_cols[1]:
            run_2cams_analysis = st.checkbox(
                "Include 2CAMS Analysis",
                value=True,
                help="Include 2CAMS data in analysis",
            )

        # Existing sections
        st.subheader("Optimization Weights")
        if single_trial:
            # Show weights for all selected movements
            for movement in selected_movements:
                st.write(f"{movement.capitalize()} weights:")
                cols = st.columns(len(weights_opt2[movement]))
                movement_weights = {}
                for i, key in enumerate(weights_opt2[movement]):
                    with cols[i]:
                        movement_weights[key] = st.slider(
                            f"{movement}_{key}",
                            min_value=0,
                            max_value=100,
                            value=weights_opt2[movement][key],
                            key=f"normal_{movement}_{key}",
                        )
                updated_weights = weights_opt2.copy()
                updated_weights[movement] = movement_weights
        else:
            # Show tabs for all movements
            movement_tabs = st.tabs(["Walking", "Squats", "STS"])
            updated_weights = {}

            for tab, (movement_key, weights) in zip(
                movement_tabs, weights_opt2.items()
            ):
                with tab:
                    st.write(f"{movement_key.capitalize()} weights:")
                    cols = st.columns(len(weights))
                    movement_weights = {}
                    for i, key in enumerate(weights):
                        with cols[i]:
                            movement_weights[key] = st.slider(
                                f"{key}",
                                min_value=0,
                                max_value=100,
                                value=weights[key],
                                key=f"normal_{movement_key}_{key}",
                            )
                    updated_weights[movement_key] = movement_weights

        weights_opt2 = updated_weights

        st.subheader("Filter Frequencies")
        if single_trial:
            # Show filter frequencies for all selected movements
            for movement in selected_movements:
                with st.columns(1)[0]:
                    filter_freq[movement] = st.slider(
                        f"{movement.capitalize()} Filter Frequency",
                        min_value=1,
                        max_value=10,
                        value=filter_freq[movement],
                        key=f"normal_filter_freq_{movement}",
                    )
        else:
            filter_freq_keys = ["walking", "squats", "STS"]
            filter_freq_cols = st.columns(len(filter_freq_keys))
            for i, key in enumerate(filter_freq_keys):
                with filter_freq_cols[i]:
                    filter_freq[key] = st.slider(
                        f"{key.capitalize()}",
                        min_value=1,
                        max_value=10,
                        value=filter_freq[key],
                        key=f"normal_filter_freq_all_{key}",
                    )

        # Add Error Thresholds section
        st.subheader("Error Thresholds")
        thresholds_cols = st.columns(len(thresholds))
        for i, key in enumerate(thresholds):
            with thresholds_cols[i]:
                if "mm" in key:
                    thresholds[key] = st.slider(
                        f"{key}",
                        min_value=0,
                        max_value=200,
                        value=thresholds[key],
                        key=f"normal_{key}",
                    )
                else:
                    thresholds[key] = st.slider(
                        f"{key}",
                        min_value=0,
                        max_value=20,
                        value=thresholds[key],
                        key=f"normal_{key}",
                    )

    # Add two columns for the action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Mono Pipeline", use_container_width=True):
            # Run the pipeline for each selected movement
            for movement in selected_movements:
                case_num, case_dir = get_or_create_case(
                    weights_opt2, filter_freq, movement
                )
                print("CASE DIR: ", case_dir)
                # trial_path = os.path.join(repo_path, "output", case_num)
                # if not os.path.exists(trial_path):
                #     os.makedirs(trial_path)
                st.write(f"Running {case_num}")

                run(
                    movement=movement,
                    estimate_local_only=estimate_local_only,
                    rerun=rerun,
                    force_rerun_analysis=force_rerun_analysis,
                    run_2cams_analysis=run_2cams_analysis,
                    output_case_path=case_dir,
                )

            sync_to_nas_if_enabled("Syncing Mono pipeline results to NAS...")

    with col2:
        if st.button("Sync Results to NAS Only", use_container_width=True):
            success = sync_to_nas_if_enabled("Syncing all results to NAS...")
            if success:
                st.success("Successfully synced results to NAS")
            elif nas_sync_available:
                st.error("Failed to sync results to NAS")
            else:
                st.warning("NAS sync module is not available. Nothing was synced.")

with hp_search_tab:
    st.header("Hyperparameter Search")
    st.write("Search for optimal weight combinations for loss terms")
    
    # Movement selection for HP search
    hp_movements = ["walking", "squats", "STS"]
    selected_hp_movements = st.multiselect(
        "Movements to search", hp_movements, default=hp_movements[:], key="hp_movement"
    )
    
    if not selected_hp_movements:
        st.warning("Please select at least one movement for hyperparameter search")
        st.stop()
    
    # Subject and session selection for HP search
    hp_subjects = ["ALL"] + [
        d
        for d in os.listdir(validation_videos_path)
        if os.path.isdir(os.path.join(validation_videos_path, d))
    ]
    selected_hp_subject = st.selectbox("Subject", hp_subjects, key="hp_subject")
    
    if selected_hp_subject != "ALL":
        subject_path = os.path.join(validation_videos_path, selected_hp_subject)
        hp_sessions = ["ALL"] + [
            d
            for d in os.listdir(os.path.join(subject_path, "VideoData"))
            if os.path.isdir(os.path.join(subject_path, "VideoData", d))
        ]
    else:
        hp_sessions = ["ALL", "Session0", "Session1"]
    selected_hp_session = st.selectbox("Session", hp_sessions, key="hp_session")
    
    # Always use Cam3 for HP search
    selected_hp_cam = "Cam3"
    st.info("📷 Hyperparameter search will use Cam3 only")
    
    # HP search settings
    st.subheader("Hyperparameter Search Settings")
    
    # Define search ranges for each weight parameter
    col1, col2 = st.columns(2)
    
    # Calculate the number of possible combinations for each movement
    def calculate_max_combinations(movement):
        current_weights = weights_opt2[movement]
        # Count parameters that are not fixed at 0
        variable_params = sum(1 for name in current_weights.keys() 
                            if name not in ['stability', 'pose_similarity'])
        # Each variable parameter has 4 possible values (1, 10, 100, 1000)
        return 4 ** variable_params
    
    # Get the maximum possible combinations across all selected movements
    max_possible_combinations = max(calculate_max_combinations(movement) 
                                  for movement in selected_hp_movements)
    
    with col1:
        st.write("**Search Ranges**")
        st.write("Values to test around current weights:")
        
        # Get current weights for reference
        current_weights = {}
        for movement in selected_hp_movements:
            current_weights[movement] = weights_opt2[movement].copy()
        
        # Display current weights
        st.write("**Current Weights:**")
        for movement in selected_hp_movements:
            st.write(f"{movement.capitalize()}: {current_weights[movement]}")
        
        st.write("**Note:** stability and pose_similarity will be fixed at 0 during search")
        
        # Show combination counts for each movement
        st.write("**Possible combinations per movement:**")
        for movement in selected_hp_movements:
            combo_count = calculate_max_combinations(movement)
            variable_params = sum(1 for name in weights_opt2[movement].keys() 
                                if name not in ['stability', 'pose_similarity'])
            st.write(f"- {movement.capitalize()}: {combo_count} combinations ({variable_params} variable parameters)")
    
    with col2:
        st.write("**Search Strategy**")
        st.write("- Test values: 1, 10, 100, 1000 for all weight terms")
        st.write("- Focus on: Mono MAE degrees, Mono MAE mm, Mono Marker MAE")
        st.write("- Marker error weighted by dividing by 10")
        st.write("- stability and pose_similarity always set to 0")
        
            # Add control for maximum combinations
    max_combinations = st.slider(
        "Max combinations per movement",
        min_value=10,
        max_value=max_possible_combinations,
        value=max_possible_combinations,
        help=f"Limit the number of combinations to test per movement (max possible: {max_possible_combinations})"
    )
    
    # Add random seed for reproducibility
    use_random_seed = st.checkbox("Use fixed random seed for reproducibility", value=True)
    if use_random_seed:
        random_seed = st.number_input("Random seed", value=42, help="Set to 0 for truly random selection")
    else:
        random_seed = None
    
    # Define search values: 1, 10, 100, 1000 for all weight terms
    
    # Generate search values based on current weights for each movement
    def generate_search_values_for_movement(movement):
        current_weights = weights_opt2[movement]
        search_values_per_weight = {}
        
        for weight_name, current_value in current_weights.items():
            # Always keep stability and pose_similarity at 0
            if weight_name in ['stability', 'pose_similarity']:
                search_values_per_weight[weight_name] = [0]
                continue
            
            # Use fixed values: 1, 10, 100, 1000
            values = [1, 10, 100, 1000]
            
            search_values_per_weight[weight_name] = values
        
        return search_values_per_weight
    
    # Load existing combinations section
    st.subheader("Load Existing Combinations")
    
    # Get list of saved combination files with status
    saved_files_with_status = get_saved_combination_files_with_status()
    saved_files_regular = get_saved_combination_files()
    
    # Combine both types of files
    all_saved_files = saved_files_with_status + saved_files_regular
    
    if all_saved_files:
        # Create a dropdown to select from saved files
        file_options = {}
        for filepath in all_saved_files:
            filename = os.path.basename(filepath)
            # Try to load metadata to show more info
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                metadata = data.get("metadata", {})
                movements_str = ", ".join(metadata.get("movements", []))
                subject = metadata.get("subject", "Unknown")
                session = metadata.get("session", "Unknown")
                total_combinations = metadata.get("total_combinations", 0)
                timestamp = metadata.get("timestamp", 0)
                date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                
                # Check if this is a status file
                has_status = "with_status" in filename
                status_indicator = "📊" if has_status else "📁"
                
                display_name = f"{status_indicator} {filename} | {movements_str} | {subject}/{session} | {total_combinations} combos | {date_str}"
                file_options[display_name] = filepath
            except:
                display_name = filename
                file_options[display_name] = filepath
        
        selected_file_display = st.selectbox(
            "Select saved combination file:",
            options=list(file_options.keys()),
            key="load_combinations_file"
        )
        
        if selected_file_display:
            selected_filepath = file_options[selected_file_display]
            
            # Show file info
            with st.expander("File Information"):
                try:
                    with open(selected_filepath, "r") as f:
                        data = json.load(f)
                    metadata = data.get("metadata", {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Movements:**", ", ".join(metadata.get("movements", [])))
                        st.write("**Subject:**", metadata.get("subject", "Unknown"))
                        st.write("**Session:**", metadata.get("session", "Unknown"))
                    
                    with col2:
                        st.write("**Max Combinations:**", metadata.get("max_combinations", "Unknown"))
                        st.write("**Random Seed:**", metadata.get("random_seed", "Unknown"))
                        st.write("**Total Combinations:**", metadata.get("total_combinations", "Unknown"))
                    
                    # Show timestamp
                    timestamp = metadata.get("timestamp", 0)
                    if timestamp:
                        date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                        st.write("**Created:**", date_str)
                    
                    # Check if this is a status file and show progress
                    has_status = "with_status" in selected_filepath
                    if has_status:
                        enhanced_combinations = data.get("combinations", {})
                        summary = get_combination_summary(enhanced_combinations)
                        
                        st.write("**📊 Progress Summary:**")
                        st.write(f"  - Total: {summary['total_combinations']}")
                        st.write(f"  - Completed: {summary['ran_combinations']}")
                        st.write(f"  - Pending: {summary['pending_combinations']}")
                        st.write(f"  - Errors: {summary['error_combinations']}")
                        
                        if summary['best_scores']:
                            st.write("**🏆 Best Scores:**")
                            for movement, score in summary['best_scores'].items():
                                st.write(f"  - {movement}: {score:.2f}")
                    else:
                        # Show combination counts per movement for regular files
                        combinations = data.get("combinations", {})
                        st.write("**Combinations per movement:**")
                        for movement, combo_list in combinations.items():
                            st.write(f"  - {movement}: {len(combo_list)} combinations")
                        
                except Exception as e:
                    st.error(f"Error reading file info: {e}")
            
            # Load button
            if st.button("Load Selected Combinations", key="load_combinations"):
                # Try to load as status file first
                combinations, metadata, run_status = load_combinations_with_status(selected_filepath)
                
                if combinations and metadata:
                    # Check compatibility with current settings
                    is_compatible, missing_movements, extra_movements = check_combinations_compatibility(
                        combinations, selected_hp_movements
                    )
                    
                    if is_compatible:
                        st.session_state.hp_combinations = combinations
                        st.session_state.loaded_metadata = metadata
                        st.session_state.run_status = run_status
                        st.session_state.current_filepath = selected_filepath
                        st.success(f"Successfully loaded combinations from {os.path.basename(selected_filepath)}")
                        st.rerun()
                    else:
                        # Show compatibility issues
                        st.warning("⚠️ Compatibility issues detected:")
                        if missing_movements:
                            st.write(f"❌ Missing movements in loaded file: {', '.join(missing_movements)}")
                        if extra_movements:
                            st.write(f"ℹ️ Extra movements in loaded file (will be ignored): {', '.join(extra_movements)}")
                        
                        # Offer to load anyway with filtered combinations
                        if st.button("Load Anyway (Filtered)", key="load_filtered"):
                            # Filter combinations to only include selected movements
                            filtered_combinations = {
                                movement: combinations[movement] 
                                for movement in selected_hp_movements 
                                if movement in combinations
                            }
                            
                            if filtered_combinations:
                                st.session_state.hp_combinations = filtered_combinations
                                st.session_state.loaded_metadata = metadata
                                st.session_state.run_status = run_status
                                st.session_state.current_filepath = selected_filepath
                                st.success(f"Successfully loaded filtered combinations from {os.path.basename(selected_filepath)}")
                                st.rerun()
                            else:
                                st.error("No compatible combinations found after filtering")
                else:
                    st.error("Failed to load combinations from file")
    else:
        st.info("No saved combination files found. Generate new combinations to save them.")
    
    st.write("---")
    
    # Generate parameter combinations
    st.subheader("Generate or Load Combinations")
    st.write("Choose how to create parameter combinations for testing:")
    st.write("- **Random Combinations**: Generate combinations using values [1, 10, 100, 1000] for each parameter")
    st.write("- **From YAML**: Load predefined combinations from `params/parameters_search.yaml`")
    
    # Show preview of parameters_search.yaml
    with st.expander("📄 Preview parameters_search.yaml"):
        params_search_path = os.path.join(repo_path, "params/parameters_search.yaml")
        if os.path.exists(params_search_path):
            try:
                # Load and display a preview
                preview_combinations = load_combinations_from_yaml(params_search_path, selected_hp_movements)
                
                if preview_combinations:
                    st.write("**Available combinations in parameters_search.yaml:**")
                    for movement, combos in preview_combinations.items():
                        st.write(f"\n**{movement.capitalize()}**: {len(combos)} combinations")
                        if combos:
                            st.write("First combination:")
                            st.json(combos[0]["weights"])
                            if len(combos) > 1:
                                with st.expander(f"Show all {len(combos)} combinations for {movement}"):
                                    for i, combo in enumerate(combos, 1):
                                        st.write(f"Combination {i}:")
                                        st.json(combo["weights"])
                else:
                    st.warning("No combinations found in parameters_search.yaml for selected movements")
            except Exception as e:
                st.error(f"Error reading parameters_search.yaml: {e}")
        else:
            st.error(f"File not found: {params_search_path}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Random Combinations", key="generate_combinations"):
            st.session_state.hp_combinations = {}
            
            for movement in selected_hp_movements:
                # Get search values for this movement
                search_values_per_weight = generate_search_values_for_movement(movement)
                weight_keys = list(weights_opt2[movement].keys())
                
                # Generate combinations using movement-specific search values
                import itertools
                search_value_lists = [search_values_per_weight[key] for key in weight_keys]
                combinations = list(itertools.product(*search_value_lists))
                
                # Convert to dictionary format
                movement_combinations = []
                for values in combinations:
                    weights_dict = {key: value for key, value in zip(weight_keys, values)}
                    movement_combinations.append({
                        "movement": movement,
                        "weights": weights_dict
                    })
                
                # Limit combinations to max_combinations
                if len(movement_combinations) > max_combinations:
                    # Randomize the order to avoid bias when taking a subset
                    if random_seed is not None and random_seed != 0:
                        random.seed(random_seed + hash(movement))  # Different seed per movement
                    random.shuffle(movement_combinations)
                    # Take a random subset of combinations
                    movement_combinations = movement_combinations[:max_combinations]
                    st.warning(f"Limited to {max_combinations} random combinations for {movement}")
                
                st.session_state.hp_combinations[movement] = movement_combinations
                st.success(f"Generated {len(movement_combinations)} combinations for {movement}")
            
            # Save generated combinations to file with status tracking
            filepath = save_combinations_with_status(st.session_state.hp_combinations, selected_hp_movements, 
                                    selected_hp_subject, selected_hp_session, max_combinations, random_seed)
            st.session_state.current_filepath = filepath
    
    with col2:
        if st.button("Load from parameters_search.yaml", key="load_from_yaml"):
            # Load combinations from parameters_search.yaml
            params_search_path = os.path.join(repo_path, "params/parameters_search.yaml")
            
            if not os.path.exists(params_search_path):
                st.error(f"File not found: {params_search_path}")
                st.stop()
            
            try:
                # Use the helper function to load combinations
                loaded_combinations = load_combinations_from_yaml(params_search_path, selected_hp_movements)
                
                if not loaded_combinations:
                    st.error("No combinations were loaded from parameters_search.yaml")
                    st.stop()
                
                st.session_state.hp_combinations = loaded_combinations
                
                # Display success message for each movement
                for movement, combos in loaded_combinations.items():
                    st.success(f"Loaded {len(combos)} combinations for {movement} from parameters_search.yaml")
                
                # Save loaded combinations to file with status tracking
                total_combinations = sum(len(combos) for combos in loaded_combinations.values())
                filepath = save_combinations_with_status(
                    st.session_state.hp_combinations, 
                    selected_hp_movements,
                    selected_hp_subject, 
                    selected_hp_session, 
                    total_combinations,
                    None  # No random seed for YAML-loaded combinations
                )
                st.session_state.current_filepath = filepath
                st.info(f"💾 Saved loaded combinations to {os.path.basename(filepath)}")
                    
            except Exception as e:
                st.error(f"Error loading parameters_search.yaml: {str(e)}")
                logger.error(f"Error loading parameters_search.yaml: {e}", exc_info=True)
    
    # Display combinations if generated or loaded
    if hasattr(st.session_state, 'hp_combinations') and st.session_state.hp_combinations:
        st.subheader("Current Combinations")
        
        # Show if combinations were loaded from file
        if hasattr(st.session_state, 'loaded_metadata'):
            metadata = st.session_state.loaded_metadata
            st.info(f"📁 Loaded from file: {metadata.get('subject', 'Unknown')}/{metadata.get('session', 'Unknown')} - {metadata.get('total_combinations', 0)} total combinations")
        
        # Show run status summary if available
        if hasattr(st.session_state, 'run_status') and st.session_state.run_status:
            st.subheader("📊 Run Status Summary")
            
            total_combinations = 0
            ran_combinations = 0
            pending_combinations = 0
            error_combinations = 0
            best_scores = {}
            
            for movement in selected_hp_movements:
                if movement in st.session_state.run_status:
                    movement_status = st.session_state.run_status[movement]
                    total_combinations += len(movement_status)
                    
                    for status in movement_status:
                        if status["ran"]:
                            ran_combinations += 1
                            if status["score"] is not None:
                                if movement not in best_scores or status["score"] < best_scores[movement]:
                                    best_scores[movement] = status["score"]
                        else:
                            pending_combinations += 1
                        
                        if status["status"] == "error":
                            error_combinations += 1
                    
                    # Show movement-specific progress
                    movement_ran = sum(1 for s in movement_status if s["ran"])
                    st.write(f"**{movement.capitalize()}**: {movement_ran}/{len(movement_status)} completed")
                    
                    if movement in best_scores:
                        st.write(f"  🏆 Best score: {best_scores[movement]:.2f}")
            
            # Overall progress
            if total_combinations > 0:
                progress_percent = (ran_combinations / total_combinations) * 100
                st.progress(ran_combinations / total_combinations)
                st.write(f"**Overall Progress**: {ran_combinations}/{total_combinations} ({progress_percent:.1f}%)")
                st.write(f"**Pending**: {pending_combinations} | **Errors**: {error_combinations}")
        
        # Show current progress if available (for backward compatibility)
        elif hasattr(st.session_state, 'hp_results') and st.session_state.hp_results:
            st.subheader("Current Progress")
            total_combinations = sum(len(combinations) for combinations in st.session_state.hp_combinations.values())
            completed_cases = 0
            
            for movement in selected_hp_movements:
                if movement in st.session_state.hp_results:
                    movement_results = st.session_state.hp_results[movement]
                    completed_cases += len(movement_results)
                    st.write(f"**{movement.capitalize()}**: {len(movement_results)}/{len(st.session_state.hp_combinations.get(movement, []))} completed")
                    
                    # Show best result so far
                    if movement_results:
                        best_result = movement_results[0]  # Results are sorted by error
                        st.write(f"  Best error: {best_result.get('total_error', 'N/A'):.2f}")
            
            if total_combinations > 0:
                progress_percent = (completed_cases / total_combinations) * 100
                st.progress(completed_cases / total_combinations)
                st.write(f"**Overall Progress**: {completed_cases}/{total_combinations} ({progress_percent:.1f}%)")
        
        for movement in selected_hp_movements:
            if movement in st.session_state.hp_combinations:
                combinations = st.session_state.hp_combinations[movement]
                st.write(f"**{movement.capitalize()}**: {len(combinations)} combinations")
                
                # Show first few combinations as example with status if available
                if len(combinations) > 0:
                    with st.expander(f"Show first 5 combinations for {movement}"):
                        for i, combo in enumerate(combinations[:5]):
                            status_info = ""
                            if hasattr(st.session_state, 'run_status') and st.session_state.run_status:
                                if movement in st.session_state.run_status and i < len(st.session_state.run_status[movement]):
                                    status = st.session_state.run_status[movement][i]
                                    if status["ran"]:
                                        status_info = f" ✅ Score: {status['score']:.2f}" if status['score'] is not None else " ✅ Completed"
                                    else:
                                        status_info = " ⏳ Pending"
                                    if status["status"] == "error":
                                        status_info = " ❌ Error"
                            
                            st.write(f"Combination {i+1}: {combo['weights']}{status_info}")
        
        # Add manual save button for loaded combinations
        if hasattr(st.session_state, 'loaded_metadata'):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Current Combinations to New File", key="save_loaded_combinations"):
                    # Get current parameters for saving
                    current_subject = selected_hp_subject
                    current_session = selected_hp_session
                    current_max_combinations = max_combinations
                    current_random_seed = random_seed
                    
                    filepath = save_combinations_to_file(
                        st.session_state.hp_combinations, 
                        selected_hp_movements,
                        current_subject, 
                        current_session, 
                        current_max_combinations, 
                        current_random_seed
                    )
                    st.success(f"Saved combinations to {os.path.basename(filepath)}")
            
            with col2:
                if st.button("Save with Status and Scores", key="save_with_status"):
                    # Get current parameters for saving
                    current_subject = selected_hp_subject
                    current_session = selected_hp_session
                    current_max_combinations = max_combinations
                    current_random_seed = random_seed
                    
                    # Get current results if available
                    current_results = st.session_state.hp_results if hasattr(st.session_state, 'hp_results') else None
                    
                    filepath = save_combinations_with_status(
                        st.session_state.hp_combinations, 
                        selected_hp_movements,
                        current_subject, 
                        current_session, 
                        current_max_combinations, 
                        current_random_seed,
                        current_results
                    )
                    st.success(f"Saved combinations with status to {os.path.basename(filepath)}")
    
    # Load existing HP search progress
    st.subheader("Resume Previous Search")
    
    progress_files = get_hp_search_progress_files()
    
    if progress_files:
        # Create a dropdown to select from progress files
        progress_options = {}
        for filepath in progress_files:
            filename = os.path.basename(filepath)
            # Try to load metadata to show more info
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                metadata = data.get("metadata", {})
                movements_str = ", ".join(metadata.get("movements", []))
                subject = metadata.get("subject", "Unknown")
                session = metadata.get("session", "Unknown")
                progress = data.get("progress", {})
                completed_cases = progress.get("completed_cases", 0)
                total_cases = progress.get("total_cases", 0)
                timestamp = data.get("timestamp", 0)
                date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                
                progress_percent = f"{completed_cases}/{total_cases}" if total_cases > 0 else "Unknown"
                display_name = f"{filename} | {movements_str} | {subject}/{session} | {progress_percent} | {date_str}"
                progress_options[display_name] = filepath
            except:
                display_name = filename
                progress_options[display_name] = filepath
        
        selected_progress_display = st.selectbox(
            "Select progress file to resume:",
            options=list(progress_options.keys()),
            key="load_progress_file"
        )
        
        if selected_progress_display:
            selected_progress_path = progress_options[selected_progress_display]
            
            # Show progress info
            with st.expander("Progress Information"):
                try:
                    with open(selected_progress_path, "r") as f:
                        data = json.load(f)
                    metadata = data.get("metadata", {})
                    progress = data.get("progress", {})
                    results = data.get("results", {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Movements:**", ", ".join(metadata.get("movements", [])))
                        st.write("**Subject:**", metadata.get("subject", "Unknown"))
                        st.write("**Session:**", metadata.get("session", "Unknown"))
                    
                    with col2:
                        completed_cases = progress.get("completed_cases", 0)
                        total_cases = progress.get("total_cases", 0)
                        st.write("**Progress:**", f"{completed_cases}/{total_cases}")
                        if total_cases > 0:
                            percent = (completed_cases / total_cases) * 100
                            st.write("**Completion:**", f"{percent:.1f}%")
                    
                    # Show timestamp
                    timestamp = data.get("timestamp", 0)
                    if timestamp:
                        date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                        st.write("**Last Updated:**", date_str)
                    
                    # Show results summary
                    if results:
                        st.write("**Current Results:**")
                        for movement, movement_results in results.items():
                            if movement_results:
                                best_result = movement_results[0]  # Results are sorted by error
                                st.write(f"  - {movement}: Best error = {best_result.get('total_error', 'N/A'):.2f}")
                        
                except Exception as e:
                    st.error(f"Error reading progress info: {e}")
            
            # Resume button
            if st.button("Resume Search", key="resume_search"):
                hp_results, combinations, metadata, progress_info = load_hp_search_progress(selected_progress_path)
                if hp_results is not None:
                    st.session_state.hp_combinations = combinations
                    st.session_state.loaded_metadata = metadata
                    st.session_state.hp_results = hp_results
                    st.session_state.progress_info = progress_info
                    st.session_state.run_status = progress_info.get("run_status", {})
                    st.success(f"Successfully loaded progress from {os.path.basename(selected_progress_path)}")
                    st.rerun()
                else:
                    st.error("Failed to load progress from file")
    else:
        st.info("No saved progress files found. Start a new search to save progress.")
    
    st.write("---")
    
    # Add a preview button to see what would be processed
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Preview Processing Plan", key="preview_plan"):
            if not hasattr(st.session_state, 'hp_combinations') or not st.session_state.hp_combinations:
                st.error("Please generate parameter combinations first")
                st.stop()
            
            st.subheader("🔍 Processing Plan Preview")
            
            # Get all videos for each movement to check existing results
            videos_by_movement = {}
            for movement in selected_hp_movements:
                videos_by_movement[movement] = get_videos_for_movement(
                    validation_videos_path, selected_hp_subject, selected_hp_session, "Cam3", movement
                )
            
            total_to_process = 0
            total_to_skip = 0
            
            for movement in selected_hp_movements:
                if movement not in st.session_state.hp_combinations:
                    continue
                    
                combinations = st.session_state.hp_combinations[movement]
                st.write(f"**{movement.capitalize()}** - {len(combinations)} combinations")
                
                movement_to_process = 0
                movement_to_skip = 0
                
                for combo_idx, combo in enumerate(combinations[:5]):  # Show first 5 as preview
                    case_num, case_dir = get_or_create_case(
                        weights_opt2, filter_freq, movement, current_params=combo
                    )
                    
                    # Check existing results
                    videos_to_process = []
                    videos_skipped = []
                    
                    for video_info in videos_by_movement[movement]:
                        result_status = check_existing_results(
                            case_dir, 
                            video_info["subject"], 
                            video_info["session"], 
                            video_info["cam"], 
                            video_info["video"], 
                            video_info["video_name"]
                        )
                        
                        if result_status["complete"]:
                            videos_skipped.append(video_info["video_name"])
                        else:
                            videos_to_process.append(video_info["video_name"])
                    
                    if videos_to_process:
                        movement_to_process += 1
                        total_to_process += 1
                        st.write(f"  Case {case_num}: {len(videos_to_process)} videos to process")
                    else:
                        movement_to_skip += 1
                        total_to_skip += 1
                        st.write(f"  Case {case_num}: All videos complete (skip)")
                
                if len(combinations) > 5:
                    st.write(f"  ... and {len(combinations) - 5} more combinations")
                
                st.write(f"  **Summary:** {movement_to_process} to process, {movement_to_skip} to skip")
                st.write("---")
            
            st.info(f"**Overall:** {total_to_process} cases to process, {total_to_skip} cases to skip")
    
    with col2:
        # Run HP search
        if st.button("Start Hyperparameter Search", key="start_hp_search"):
            if not hasattr(st.session_state, 'hp_combinations') or not st.session_state.hp_combinations:
                st.error("Please generate parameter combinations first")
                st.stop()
            
            # Check if we have status tracking and show unfinished combinations
            if hasattr(st.session_state, 'run_status') and st.session_state.run_status:
                unfinished_combinations = {}
                total_unfinished = 0
                
                for movement in selected_hp_movements:
                    if movement in st.session_state.run_status:
                        unfinished = []
                        for i, status in enumerate(st.session_state.run_status[movement]):
                            if not status["ran"]:
                                unfinished.append(st.session_state.hp_combinations[movement][i])
                        unfinished_combinations[movement] = unfinished
                        total_unfinished += len(unfinished)
                
                if total_unfinished > 0:
                    st.info(f"📊 Found {total_unfinished} unfinished combinations to process")
                    for movement, unfinished in unfinished_combinations.items():
                        if unfinished:
                            st.write(f"  - {movement}: {len(unfinished)} combinations remaining")
                else:
                    st.success("🎉 All combinations have been processed!")
                    st.stop()
            
            # Initialize or load existing results
            if hasattr(st.session_state, 'hp_results'):
                hp_results = st.session_state.hp_results
                st.info("📁 Resuming from previous progress...")
            else:
                hp_results = {}
            
            # Initialize or load existing progress
            if hasattr(st.session_state, 'progress_info'):
                progress_info = st.session_state.progress_info
                current_combination = progress_info.get("completed_cases", 0)
                st.info(f"📈 Resuming from case {current_combination + 1}")
            else:
                progress_info = {"completed_cases": 0, "total_cases": 0}
                current_combination = 0
            
            # Get all videos for each movement to check existing results
            st.write("🔍 Checking existing results...")
            videos_by_movement = {}
            for movement in selected_hp_movements:
                videos_by_movement[movement] = get_videos_for_movement(
                    validation_videos_path, selected_hp_subject, selected_hp_session, "Cam3", movement
                )
                st.write(f"Found {len(videos_by_movement[movement])} videos for {movement}")
            
            # Progress tracking
            if hasattr(st.session_state, 'run_status') and st.session_state.run_status:
                # Use unfinished combinations for progress tracking
                total_combinations = 0
                for movement in selected_hp_movements:
                    if movement in st.session_state.run_status:
                        total_combinations += sum(1 for status in st.session_state.run_status[movement] if not status["ran"])
            else:
                total_combinations = sum(len(combinations) for combinations in st.session_state.hp_combinations.values())
            
            progress_info["total_cases"] = total_combinations
            progress_bar = st.progress(current_combination / total_combinations if total_combinations > 0 else 0)
            status_text = st.empty()
            
            # Store results for each movement
            
            for movement in selected_hp_movements:
                if movement not in st.session_state.hp_combinations:
                    continue
                
                # Use unfinished combinations if status tracking is available
                if hasattr(st.session_state, 'run_status') and st.session_state.run_status:
                    if movement in st.session_state.run_status:
                        # Get unfinished combinations
                        unfinished_indices = []
                        for i, status in enumerate(st.session_state.run_status[movement]):
                            if not status["ran"]:
                                unfinished_indices.append(i)
                        
                        if not unfinished_indices:
                            st.write(f"✅ All combinations for {movement} are already completed")
                            continue
                        
                        combinations = [st.session_state.hp_combinations[movement][i] for i in unfinished_indices]
                        st.write(f"🔄 Processing {len(combinations)} unfinished combinations for {movement}")
                    else:
                        combinations = st.session_state.hp_combinations[movement]
                else:
                    combinations = st.session_state.hp_combinations[movement]
                
                movement_results = []
                
                # Track original indices for status updates
                original_indices = []
                if hasattr(st.session_state, 'run_status') and st.session_state.run_status:
                    if movement in st.session_state.run_status:
                        # Get unfinished combinations with their original indices
                        unfinished_indices = []
                        for i, status in enumerate(st.session_state.run_status[movement]):
                            if not status["ran"]:
                                unfinished_indices.append(i)
                        original_indices = unfinished_indices
                    else:
                        original_indices = list(range(len(combinations)))
                else:
                    original_indices = list(range(len(combinations)))
                
                for combo_idx, combo in enumerate(combinations):
                    # Check if this combination was already processed using run status
                    already_processed = False
                    if hasattr(st.session_state, 'run_status') and st.session_state.run_status:
                        if movement in st.session_state.run_status and combo_idx < len(st.session_state.run_status[movement]):
                            if st.session_state.run_status[movement][combo_idx]["ran"]:
                                already_processed = True
                    
                    # Also check existing results for backward compatibility
                    if not already_processed and movement in hp_results:
                        for existing_result in hp_results[movement]:
                            if (existing_result.get("parameters", {}).get("weights") == combo["weights"] and
                                existing_result.get("parameters", {}).get("movement") == combo["movement"]):
                                already_processed = True
                                break
                    
                    if already_processed:
                        current_combination += 1
                        progress = current_combination / total_combinations
                        progress_bar.progress(progress)
                        status_text.text(f"Skipping already processed {movement} combination {combo_idx + 1}/{len(combinations)}")
                        continue
                    
                    current_combination += 1
                    progress = current_combination / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {movement} combination {combo_idx + 1}/{len(combinations)}")
                    
                    # Create case for this combination
                    case_num, case_dir = get_or_create_case(
                        weights_opt2, filter_freq, movement, current_params=combo
                    )
                    
                    st.write(f"Checking case {case_num} for {movement}")
                    
                    # Check existing results for all videos in this movement
                    videos_to_process = []
                    videos_skipped = []
                    videos_partial = []
                    
                    for video_info in videos_by_movement[movement]:
                        result_status = check_existing_results(
                            case_dir, 
                            video_info["subject"], 
                            video_info["session"], 
                            video_info["cam"], 
                            video_info["video"], 
                            video_info["video_name"]
                        )
                        
                        if result_status["complete"]:
                            videos_skipped.append(video_info["video_name"])
                        elif result_status["partial"]:
                            videos_partial.append(video_info["video_name"])
                            videos_to_process.append(video_info)
                        else:
                            videos_to_process.append(video_info)
                    
                    # Display status
                    if videos_skipped:
                        st.write(f"✅ Skipping {len(videos_skipped)} videos with complete results")
                    if videos_partial:
                        st.write(f"⚠️ Found {len(videos_partial)} videos with partial results")
                    if videos_to_process:
                        st.write(f"🔄 Processing {len(videos_to_process)} videos:")
                        for video_info in videos_to_process:
                            st.write(f"   - {video_info['subject']}/{video_info['session']}/{video_info['cam']}/{video_info['video']}/{video_info['video_name']}")
                    
                    # If all videos are complete, skip this combination entirely
                    if not videos_to_process:
                        st.write(f"🎉 All videos for case {case_num} already have complete results. Skipping...")
                        
                        # Aggregate results from all translation_error.csv files
                        aggregated = aggregate_case_results(case_dir)
                        
                        if aggregated is not None:
                            avg_mae_mm = aggregated['avg_mae_mm']
                            avg_mae_degrees = aggregated['avg_mae_degrees']
                            marker_mae = aggregated['marker_mae']
                            foot_marker_error = aggregated['foot_marker_error']
                            num_videos = aggregated['num_videos']
                            
                            # Calculate weighted total error
                            total_error = avg_mae_mm + avg_mae_degrees + (marker_mae / 10) + (foot_marker_error / 5)

                            # Display results
                            st.write(f"📊 Aggregated results from {num_videos} videos:")
                            st.write(f"avg_mae_mm MAE: {avg_mae_mm:.2f}")
                            st.write(f"avg_mae_degrees: {avg_mae_degrees:.2f}")
                            st.write(f"marker_mae: {marker_mae:.2f}")
                            st.write(f"foot_marker_error: {foot_marker_error:.2f}")
                     
                            result = {
                                "case_num": case_num,
                                "parameters": combo,
                                "avg_mae_mm": avg_mae_mm,
                                "avg_mae_degrees": avg_mae_degrees,
                                "marker_mae": marker_mae,
                                "foot_marker_error": foot_marker_error,
                                "total_error": total_error,
                                "output_csv": case_dir,  # Store case directory instead of single file
                                "num_videos_aggregated": num_videos,
                                "status": "skipped_complete",
                                "videos_skipped": videos_skipped,
                                "videos_processed": [],
                                "videos_partial": videos_partial
                            }
                            
                            movement_results.append(result)
                            st.write(f"📊 Loaded existing results: MAE mm: {avg_mae_mm:.2f}, MAE degrees: {avg_mae_degrees:.2f}, Foot MAE: {foot_marker_error:.2f}, Total Error: {total_error:.2f}")
                            
                            # Update status in saved file if available
                            if hasattr(st.session_state, 'current_filepath') and st.session_state.current_filepath:
                                original_idx = original_indices[combo_idx] if combo_idx < len(original_indices) else combo_idx
                                update_combination_status(
                                    st.session_state.current_filepath, 
                                    movement, 
                                    original_idx, 
                                    ran=True, 
                                    score=total_error, 
                                    status="skipped_complete"
                                )
                            
                            continue
                        else:
                            st.warning(f"Could not aggregate results for case {case_num} - no valid CSV files found")
                            # Continue with processing
                    
                    try:
                        # Check if WHAM files already exist to skip processing
                        wham_files_exist = False
                        if selected_hp_subject != "ALL":
                            # Check if WHAM output exists for this subject/Cam3 combination
                            wham_check_path = os.path.join(
                                repo_path, "output", selected_hp_subject, 
                                selected_hp_session if selected_hp_session != "ALL" else "Session0",
                                "Cam3", "wham_output.pkl"
                            )
                            wham_files_exist = os.path.exists(wham_check_path)
                        
                        # Run the pipeline with current parameters (always using Cam3)
                        # Only process videos that need processing
                        if videos_to_process:
                            st.write(f"Running case {case_num} for {movement} with {len(videos_to_process)} videos")
                            
                            output_csv_path = run(
                                movement=movement,
                                test_subjects=[selected_hp_subject] if selected_hp_subject != "ALL" else None,
                                test_cameras=["Cam3"],  # Always use Cam3 for HP search
                                estimate_local_only=True,
                                rerun=False,  # Don't rerun WHAM
                                force_rerun_analysis=False,
                                run_2cams_analysis=True,
                                output_case_path=case_dir,
                                success_message=f"Completed case {case_num}",
                                current_params=combo,  # Pass current parameters for HP search
                                specific_videos=videos_to_process,  # Only process videos that need processing
                            )
                        
                        # Aggregate results from ALL translation_error.csv files in the case directory
                        # This properly combines results from all processed videos
                        aggregated = aggregate_case_results(case_dir)
                        
                        if aggregated is not None:
                            avg_mae_mm = aggregated['avg_mae_mm']
                            avg_mae_degrees = aggregated['avg_mae_degrees']
                            marker_mae = aggregated['marker_mae']
                            foot_marker_error = aggregated['foot_marker_error']
                            num_videos = aggregated['num_videos']
                            
                            # Calculate weighted total error (marker error divided by 10, foot error divided by 5)
                            total_error = avg_mae_mm + avg_mae_degrees + (marker_mae / 10) + (foot_marker_error / 5)

                            # Display results
                            st.write(f"📊 Aggregated results from {num_videos} videos:")
                            st.write(f"avg_mae_mm MAE: {avg_mae_mm:.2f}")
                            st.write(f"avg_mae_degrees: {avg_mae_degrees:.2f}")
                            st.write(f"marker_mae: {marker_mae:.2f}")
                            st.write(f"foot_marker_error: {foot_marker_error:.2f}")
                            st.write(f"total_error: {total_error:.2f}")
                            
                            result = {
                                "case_num": case_num,
                                "parameters": combo,
                                "avg_mae_mm": avg_mae_mm,
                                "avg_mae_degrees": avg_mae_degrees,
                                "marker_mae": marker_mae,
                                "foot_marker_error": foot_marker_error,
                                "total_error": total_error,
                                "output_csv": case_dir,  # Store case directory instead of single file
                                "num_videos_aggregated": num_videos,
                                "status": "processed",
                                "videos_skipped": videos_skipped,
                                "videos_processed": [v["video_name"] for v in videos_to_process] if videos_to_process else [],
                                "videos_partial": videos_partial
                            }
                            
                            movement_results.append(result)
                            
                            st.write(f"Case {case_num} - MAE mm: {avg_mae_mm:.2f}, MAE degrees: {avg_mae_degrees:.2f}, Marker MAE: {marker_mae:.2f}, Foot MAE: {foot_marker_error:.2f}, Total Error: {total_error:.2f}")
                            
                            # Update status in saved file if available
                            if hasattr(st.session_state, 'current_filepath') and st.session_state.current_filepath:
                                original_idx = original_indices[combo_idx] if combo_idx < len(original_indices) else combo_idx
                                update_combination_status(
                                    st.session_state.current_filepath, 
                                    movement, 
                                    original_idx, 
                                    ran=True, 
                                    score=total_error, 
                                    status="processed"
                                )
                        else:
                            st.error(f"No valid results found in {case_dir}")
                        
                    except Exception as e:
                        # Get detailed error information
                        error_details = format_exception_details(
                            e, 
                            context=f"Processing case {case_num} for movement {movement}"
                        )
                        
                        # Display error in Streamlit
                        display_error_in_streamlit(error_details, case_num=case_num)
                        
                        # Log the error
                        log_error_details(error_details, logger, case_num=case_num)
                        
                        # Add the failed case to results with high error
                        result = {
                            "case_num": case_num,
                            "parameters": combo,
                            "avg_mae_mm": 999.0,  # High error to rank it last
                            "avg_mae_degrees": 999.0,
                            "marker_mae": 999.0,
                            "total_error": 999.0,
                            "output_csv": None,
                            "error": error_details["message"],
                            "error_type": error_details["type"],
                            "error_location": error_details["location"],
                            "error_line": error_details["line"],
                            "traceback": error_details["traceback"],
                            "status": "error",
                            "videos_skipped": videos_skipped,
                            "videos_processed": [v["video_name"] for v in videos_to_process],
                            "videos_partial": videos_partial
                        }
                        movement_results.append(result)
                        
                        # Update status in saved file if available
                        if hasattr(st.session_state, 'current_filepath') and st.session_state.current_filepath:
                            original_idx = original_indices[combo_idx] if combo_idx < len(original_indices) else combo_idx
                            update_combination_status(
                                st.session_state.current_filepath, 
                                movement, 
                                original_idx, 
                                ran=True, 
                                score=999.0, 
                                error=str(e),
                                status="error"
                            )
                        
                        continue
                    
                    # Update progress and save periodically
                    progress_info["completed_cases"] = current_combination
                    
                    # Save progress every 5 cases or at the end
                    if current_combination % 5 == 0 or current_combination == total_combinations:
                        try:
                            # Get metadata for saving
                            if hasattr(st.session_state, 'loaded_metadata'):
                                metadata = st.session_state.loaded_metadata
                            else:
                                metadata = {
                                    "movements": selected_hp_movements,
                                    "subject": selected_hp_subject,
                                    "session": selected_hp_session,
                                    "max_combinations": max_combinations,
                                    "random_seed": random_seed
                                }
                            
                            save_hp_search_progress(hp_results, st.session_state.hp_combinations, metadata, progress_info)
                            st.write(f"💾 Progress saved at case {current_combination}")
                        except Exception as save_error:
                            st.warning(f"Could not save progress: {save_error}")
                    
                                    # Sort results by total error (lowest first)
                movement_results.sort(key=lambda x: x["total_error"])
                hp_results[movement] = movement_results
            
            # Save final results
            results_file = os.path.join(repo_path, "output", "hp_search_results.json")
            with open(results_file, "w") as f:
                json.dump(hp_results, f, indent=4, default=str)
            
            # Save final progress
            try:
                if hasattr(st.session_state, 'loaded_metadata'):
                    metadata = st.session_state.loaded_metadata
                else:
                    metadata = {
                        "movements": selected_hp_movements,
                        "subject": selected_hp_subject,
                        "session": selected_hp_session,
                        "max_combinations": max_combinations,
                        "random_seed": random_seed
                    }
                
                progress_info["completed_cases"] = total_combinations
                save_hp_search_progress(hp_results, st.session_state.hp_combinations, metadata, progress_info)
                st.success("💾 Final progress saved")
            except Exception as save_error:
                st.warning(f"Could not save final progress: {save_error}")
            
            # Display summary of what was processed vs skipped
            st.success("Hyperparameter search completed!")
            
            # Summary statistics
            total_cases = 0
            skipped_cases = 0
            processed_cases = 0
            
            for movement in selected_hp_movements:
                if movement in hp_results:
                    for result in hp_results[movement]:
                        total_cases += 1
                        if result.get("status") == "skipped_complete":
                            skipped_cases += 1
                        else:
                            processed_cases += 1
            
            st.subheader("📊 Processing Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cases", total_cases)
            with col2:
                st.metric("Skipped (Complete)", skipped_cases)
            with col3:
                st.metric("Processed", processed_cases)
            
            if skipped_cases > 0:
                st.info(f"🎉 Skipped {skipped_cases} cases that already had complete results, saving significant processing time!")
            
            # Display best results
            st.subheader("Best Results by Movement")
            for movement in selected_hp_movements:
                st.write(f"\n### {movement.capitalize()}")

                results = analyze_hp_results(movement)
                if results is None:
                    st.write(f"No results found for {movement}")
                    continue

                # Get top 3 combinations
                top_3 = dict(list(results.items())[:3])

                for rank, (case_num, data) in enumerate(top_3.items(), 1):
                    st.write(f"\n**Rank {rank}** (Case: {case_num})")

                    # Create a formatted display of the results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Parameters:")
                        st.json(data["parameters"])

                    with col2:
                        st.write("Results:")
                        metrics = {
                            "Average MAE (mm)": f"{data['avg_mae_mm']:.2f}",
                            "Average MAE (degrees)": f"{data['avg_mae_degrees']:.2f}",
                            "Total Error": f"{data['total_error']:.2f}",
                        }
                        for metric, value in metrics.items():
                            st.write(f"{metric}: {value}")

                    st.write("---")
            
            # Create summary table
            st.subheader("Summary Table")
            summary_data = []
            for movement in selected_hp_movements:
                if movement in hp_results and hp_results[movement]:
                    # Get top 3 results
                    top_3 = hp_results[movement][:3]
                    for i, result in enumerate(top_3):
                        summary_data.append({
                            "Movement": movement.capitalize(),
                            "Rank": i + 1,
                            "Case": result['case_num'],
                            "Total Error": f"{result['total_error']:.2f}",
                            "MAE mm": f"{result['avg_mae_mm']:.2f}",
                            "MAE degrees": f"{result['avg_mae_degrees']:.2f}",
                            "Marker MAE": f"{result['marker_mae']:.2f}",
                            "Reprojection": result['parameters']['weights'].get('reprojection', 'N/A'),
                            "Contact Position": result['parameters']['weights'].get('contact_position', 'N/A'),
                            "Flat Floor": result['parameters']['weights'].get('flat_floor', 'N/A'),
                            "Stability": "0 (fixed)",
                            "Pose Similarity": "0 (fixed)",
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            
            # Add detailed view of video processing
            if st.checkbox("Show detailed video processing information", value=False):
                st.subheader("📋 Detailed Video Processing")
                
                for movement in selected_hp_movements:
                    if movement in hp_results and hp_results[movement]:
                        st.write(f"**{movement.capitalize()}**")
                        
                        for result in hp_results[movement]:
                            with st.expander(f"Case {result['case_num']} - {result['status']}"):
                                st.write(f"**Status:** {result['status']}")
                                st.write(f"**Total Error:** {result['total_error']:.2f}")
                                
                                if result.get('videos_skipped'):
                                    st.write(f"**Skipped videos ({len(result['videos_skipped'])}):**")
                                    for video in result['videos_skipped']:
                                        st.write(f"  ✅ {video}")
                                
                                if result.get('videos_processed'):
                                    st.write(f"**Processed videos ({len(result['videos_processed'])}):**")
                                    for video in result['videos_processed']:
                                        st.write(f"  🔄 {video}")
                                
                                if result.get('videos_partial'):
                                    st.write(f"**Partial results ({len(result['videos_partial'])}):**")
                                    for video in result['videos_partial']:
                                        st.write(f"  ⚠️ {video}")
                                
                                if result.get('error'):
                                    st.error(f"**Error:** {result['error']}")
                        
                        st.write("---")
            
            # Option to update parameters.yaml with best results
            if st.button("Update Parameters with Best Results", key="update_params"):
                try:
                    # Load current parameters
                    with open(params_path, "r") as f:
                        current_params = yaml.safe_load(f)
                    
                    # Update with best results
                    for movement in selected_hp_movements:
                        if movement in hp_results and hp_results[movement]:
                            best_result = hp_results[movement][0]
                            best_weights = best_result['parameters']['weights']
                            
                            # Update the corresponding weights in the parameters
                            if movement == "walking":
                                current_params["weights_opt2_walking"] = best_weights
                            elif movement == "squats":
                                current_params["weights_opt2_squats"] = best_weights
                            elif movement == "STS":
                                current_params["weights_opt2_sts"] = best_weights
                    
                    # Save updated parameters
                    with open(params_path, "w") as f:
                        yaml.dump(current_params, f, default_flow_style=False, indent=2)
                    
                    st.success("Parameters updated with best hyperparameter search results!")
                    
                except Exception as e:
                    st.error(f"Error updating parameters: {str(e)}")
    
    # Display existing results
    if st.button("Load Existing Results", key="load_results"):
        results_file = os.path.join(repo_path, "output", "hp_search_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                existing_results = json.load(f)
            
            st.subheader("Existing Hyperparameter Search Results")
            
            for movement in selected_hp_movements:
                if movement in existing_results and existing_results[movement]:
                    st.write(f"**{movement.capitalize()}**")
                    
                    # Show top 5 results
                    top_5 = existing_results[movement][:5]
                    
                    for i, result in enumerate(top_5):
                        st.write(f"**Rank {i+1}**: {result['case_num']}")
                        st.write(f"Total Error: {result['total_error']:.2f}")
                        st.write(f"MAE mm: {result['avg_mae_mm']:.2f}, MAE degrees: {result['avg_mae_degrees']:.2f}")
                        if result.get('marker_mae'):
                            st.write(f"Marker MAE: {result['marker_mae']:.2f}")
                        st.write("Parameters:")
                        st.json(result['parameters'])
                        st.write("---")
        else:
            st.warning("No existing results found")