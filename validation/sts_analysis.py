import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import json
from sts_segmentor import segment_STS, storage_to_numpy
from scipy.spatial.distance import euclidean

def analyze_sts_trials(base_folders, output_dir=None, visualize=True, analysis_type='discrete'):
    """
    Walk through the specified folders to find STS trials, segment them using segment_STS,
    and save the segmentation results.
    
    Args:
        base_folders (list): List of base folder paths to search
        output_dir (str, optional): Directory to save summary output files
        visualize (bool): Whether to visualize results during segmentation
        analysis_type (str): Type of pelvis shift analysis - 'discrete' or 'continuous'
    """
    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Summary files will be saved to: {os.path.abspath(output_dir)}")
    
    # Create a log file for detailed tracking
    log_path = os.path.join(output_dir, "sts_analysis_log.txt")
    with open(log_path, 'w') as log_file:
        log_file.write("=== STS Analysis Log ===\n\n")
        log_file.write(f"Analysis started at: {pd.Timestamp.now()}\n\n")
        log_file.write("Files processed:\n\n")
    
    # Lists to store processed file paths and segmentation results
    processed_files = []
    segmentation_summary = []
    
    # Walk through each base folder
    for base_folder in base_folders:
        print(f"Searching in {base_folder}...")
        
        for root, dirs, files in os.walk(base_folder):
            # Check if we're in the OpenSim/IK/shiftedIK directory
            if 'OpenSim/IK/shiftedIK' in root:
                # Look for mot files (both mono and wham versions)
                mono_mot_files = [f for f in files if f.endswith('.mot') and not f.endswith('_wham.mot')]
                wham_mot_files = [f for f in files if f.endswith('sync_wham.mot')]
                
                # Extract metadata from path
                subject_match = re.search(r'subject(\d+)', root)
                subject = f"subject{subject_match.group(1)}" if subject_match else "Unknown"
                
                camera_match = re.search(r'Cam(\d+)', root)
                camera = camera_match.group(1) if camera_match else "Unknown"
                
                # Extract trial information from path
                trial_parts = root.split(os.sep)
                trial = None
                for part in trial_parts:
                    if part.startswith('STS'):
                        trial = part
                        break
                
                if trial is None:
                    continue  # Skip if not an STS trial
                
                print(f"Found STS trial: Subject {subject}, Camera {camera}, Trial {trial}")
                
                # Construct path to the mocap file (up from shiftedIK folder)
                trial_folder = None
                for i, part in enumerate(trial_parts):
                    if part == trial:
                        trial_folder = os.sep.join(trial_parts[:i+1])
                        break
                
                mocap_file = None
                if trial_folder:
                    mocap_path = os.path.join(trial_folder, 'mocap', f"{trial}.mot")
                    if os.path.exists(mocap_path):
                        mocap_file = mocap_path
                        print(f"Found mocap file: {mocap_path}")
                
                # Find 2-camera file
                twocam_file = None
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                twocam_folder = os.path.join(
                    project_root,
                    f'LabValidation_withVideos1/{subject}/OpenSimData/Video/HRNet/2-cameras/IK'
                )
                trial_name = trial.split('_')[0] if '_' in trial else trial
                twocam_path = os.path.join(twocam_folder, f'{trial_name}.mot')
                if os.path.exists(twocam_path):
                    twocam_file = twocam_path
                    print(f"Found 2-camera file: {twocam_path}")
                
                # Process mono and wham mot files
                for mono_file in mono_mot_files:
                    mono_path = os.path.join(root, mono_file)
                    
                    # Find corresponding wham file
                    wham_file = f"{os.path.splitext(mono_file)[0]}_wham.mot"
                    wham_path = os.path.join(root, wham_file)
                    
                    if os.path.exists(wham_path):
                        try:
                            print(f"Processing: {mono_file}")
                            
                            # Log the files
                            with open(log_path, 'a') as log_file:
                                log_file.write(f"Processing trial: {trial}\n")
                                log_file.write(f"  Subject: {subject}, Camera: {camera}\n")
                                log_file.write(f"  Mono file: {mono_path}\n")
                                log_file.write(f"  WHAM file: {wham_path}\n")
                                if mocap_file:
                                    log_file.write(f"  Mocap file: {mocap_file}\n")
                                if twocam_file:
                                    log_file.write(f"  2-Camera file: {twocam_file}\n")
                            
                            # Segment the mono file
                            mono_rising_times, mono_rising_delayed, mono_rising_sitting = segment_STS(
                                mono_path, visualize=visualize)
                            
                            # Segment the wham file
                            wham_rising_times, wham_rising_delayed, wham_rising_sitting = segment_STS(
                                wham_path, visualize=visualize)
                            
                            # Segment the mocap file if available
                            mocap_rising_times, mocap_rising_delayed, mocap_rising_sitting = None, None, None
                            if mocap_file:
                                try:
                                    mocap_rising_times, mocap_rising_delayed, mocap_rising_sitting = segment_STS(
                                        mocap_file, visualize=visualize)
                                except Exception as e:
                                    print(f"Error segmenting mocap file: {str(e)}")
                                    with open(log_path, 'a') as log_file:
                                        log_file.write(f"  ERROR segmenting mocap: {str(e)}\n")
                            
                            # Segment the 2-camera file if available
                            twocam_rising_times, twocam_rising_delayed, twocam_rising_sitting = None, None, None
                            if twocam_file:
                                try:
                                    twocam_rising_times, twocam_rising_delayed, twocam_rising_sitting = segment_STS(
                                        twocam_file, visualize=visualize)
                                except Exception as e:
                                    print(f"Error segmenting 2-camera file: {str(e)}")
                                    with open(log_path, 'a') as log_file:
                                        log_file.write(f"  ERROR segmenting 2-camera: {str(e)}\n")
                            
                            # Save segmentation results
                            results_dir = os.path.join(root, 'SegmentationResults')
                            os.makedirs(results_dir, exist_ok=True)
                            
                            # Create base filenames
                            base_mono = os.path.splitext(mono_file)[0]
                            base_wham = os.path.splitext(wham_file)[0]
                            
                            # Save mono results
                            mono_results = {
                                'risingTimes': mono_rising_times,
                                'risingTimesDelayedStart': mono_rising_delayed,
                                'risingSittingTimesDelayedStartPeriodicEnd': mono_rising_sitting
                            }
                            
                            mono_json_path = os.path.join(results_dir, f"{base_mono}_segments.json")
                            with open(mono_json_path, 'w') as f:
                                json.dump(mono_results, f, indent=4)
                            
                            # Save wham results
                            wham_results = {
                                'risingTimes': wham_rising_times,
                                'risingTimesDelayedStart': wham_rising_delayed,
                                'risingSittingTimesDelayedStartPeriodicEnd': wham_rising_sitting
                            }
                            
                            wham_json_path = os.path.join(results_dir, f"{base_wham}_segments.json")
                            with open(wham_json_path, 'w') as f:
                                json.dump(wham_results, f, indent=4)
                            
                            # Save mocap results if available
                            if mocap_file and mocap_rising_times:
                                mocap_base = os.path.splitext(os.path.basename(mocap_file))[0]
                                mocap_results = {
                                    'risingTimes': mocap_rising_times,
                                    'risingTimesDelayedStart': mocap_rising_delayed,
                                    'risingSittingTimesDelayedStartPeriodicEnd': mocap_rising_sitting
                                }
                                
                                mocap_json_path = os.path.join(results_dir, f"{mocap_base}_segments.json")
                                with open(mocap_json_path, 'w') as f:
                                    json.dump(mocap_results, f, indent=4)
                            
                            # Also save as CSV for easier analysis
                            # Convert the list of lists to DataFrame
                            mono_df = pd.DataFrame({
                                'segment': range(1, len(mono_rising_times) + 1),
                                'rising_start': [t[0] for t in mono_rising_times],
                                'rising_end': [t[1] for t in mono_rising_times],
                                'delayed_start': [t[0] for t in mono_rising_delayed],
                                'delayed_end': [t[1] for t in mono_rising_delayed],
                                'periodic_start': [t[0] for t in mono_rising_sitting],
                                'periodic_end': [t[1] for t in mono_rising_sitting]
                            })
                            
                            mono_csv_path = os.path.join(results_dir, f"{base_mono}_segments.csv")
                            mono_df.to_csv(mono_csv_path, index=False)
                            
                            wham_df = pd.DataFrame({
                                'segment': range(1, len(wham_rising_times) + 1),
                                'rising_start': [t[0] for t in wham_rising_times],
                                'rising_end': [t[1] for t in wham_rising_times],
                                'delayed_start': [t[0] for t in wham_rising_delayed],
                                'delayed_end': [t[1] for t in wham_rising_delayed],
                                'periodic_start': [t[0] for t in wham_rising_sitting],
                                'periodic_end': [t[1] for t in wham_rising_sitting]
                            })
                            
                            wham_csv_path = os.path.join(results_dir, f"{base_wham}_segments.csv")
                            wham_df.to_csv(wham_csv_path, index=False)
                            
                            # Save mocap CSV if available
                            if mocap_file and mocap_rising_times:
                                mocap_df = pd.DataFrame({
                                    'segment': range(1, len(mocap_rising_times) + 1),
                                    'rising_start': [t[0] for t in mocap_rising_times],
                                    'rising_end': [t[1] for t in mocap_rising_times],
                                    'delayed_start': [t[0] for t in mocap_rising_delayed],
                                    'delayed_end': [t[1] for t in mocap_rising_delayed],
                                    'periodic_start': [t[0] for t in mocap_rising_sitting],
                                    'periodic_end': [t[1] for t in mocap_rising_sitting]
                                })
                                
                                mocap_csv_path = os.path.join(results_dir, f"{mocap_base}_segments.csv")
                                mocap_df.to_csv(mocap_csv_path, index=False)
                            
                            # Save 2-camera results if available
                            if twocam_file and twocam_rising_times:
                                twocam_base = os.path.splitext(os.path.basename(twocam_file))[0]
                                twocam_results = {
                                    'risingTimes': twocam_rising_times,
                                    'risingTimesDelayedStart': twocam_rising_delayed,
                                    'risingSittingTimesDelayedStartPeriodicEnd': twocam_rising_sitting
                                }
                                
                                twocam_json_path = os.path.join(results_dir, f"{twocam_base}_twocam_segments.json")
                                with open(twocam_json_path, 'w') as f:
                                    json.dump(twocam_results, f, indent=4)
                                
                                twocam_df = pd.DataFrame({
                                    'segment': range(1, len(twocam_rising_times) + 1),
                                    'rising_start': [t[0] for t in twocam_rising_times],
                                    'rising_end': [t[1] for t in twocam_rising_times],
                                    'delayed_start': [t[0] for t in twocam_rising_delayed],
                                    'delayed_end': [t[1] for t in twocam_rising_delayed],
                                    'periodic_start': [t[0] for t in twocam_rising_sitting],
                                    'periodic_end': [t[1] for t in twocam_rising_sitting]
                                })
                                
                                twocam_csv_path = os.path.join(results_dir, f"{twocam_base}_twocam_segments.csv")
                                twocam_df.to_csv(twocam_csv_path, index=False)
                            
                            # Add to summary
                            summary_entry = {
                                'Subject': subject,
                                'Camera': camera,
                                'Trial': trial,
                                'MonoFile': mono_file,
                                'WhamFile': wham_file,
                                'MonoSegments': len(mono_rising_times),
                                'WhamSegments': len(wham_rising_times)
                            }
                            
                            if mocap_file and mocap_rising_times:
                                summary_entry['MocapFile'] = os.path.basename(mocap_file)
                                summary_entry['MocapSegments'] = len(mocap_rising_times)
                            
                            if twocam_file and twocam_rising_times:
                                summary_entry['TwoCamFile'] = os.path.basename(twocam_file)
                                summary_entry['TwoCamSegments'] = len(twocam_rising_times)
                            
                            segmentation_summary.append(summary_entry)
                            
                            processed_files.append((mono_path, wham_path, mocap_file if mocap_file else None, twocam_file if twocam_file else None))
                            
                            # Log success
                            with open(log_path, 'a') as log_file:
                                log_file.write(f"  Segmentation successful\n")
                                log_file.write(f"  Mono segments: {len(mono_rising_times)}\n")
                                log_file.write(f"  WHAM segments: {len(wham_rising_times)}\n")
                                if mocap_file and mocap_rising_times:
                                    log_file.write(f"  Mocap segments: {len(mocap_rising_times)}\n")
                                if twocam_file and twocam_rising_times:
                                    log_file.write(f"  2-Camera segments: {len(twocam_rising_times)}\n")
                                log_file.write(f"  Results saved to: {results_dir}\n\n")
                            
                            print(f"  Segmentation results saved to {results_dir}")
                            
                            # Create comparison visualization
                            create_comparison_plot(
                                mono_path, wham_path, mocap_file,
                                mono_rising_sitting, wham_rising_sitting, mocap_rising_sitting,
                                results_dir, base_mono, analysis_type,
                                twocam_path=twocam_file, twocam_segments=twocam_rising_sitting
                            )
                            
                        except Exception as e:
                            print(f"Error processing files {mono_file}/{wham_file}: {str(e)}")
                            with open(log_path, 'a') as log_file:
                                log_file.write(f"  ERROR: {str(e)}\n\n")
    
    # Save summary to CSV
    summary_df = pd.DataFrame(segmentation_summary)
    if not summary_df.empty:
        summary_path = os.path.join(output_dir, "sts_segmentation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {os.path.abspath(summary_path)}")
    
    # Log final stats
    with open(log_path, 'a') as log_file:
        log_file.write("\n=== Analysis Summary ===\n\n")
        log_file.write(f"Total trials processed: {len(processed_files)}\n")
    
    print(f"\nProcessed {len(processed_files)} STS trials")
    print(f"Detailed log written to {os.path.abspath(log_path)}")
    
    return processed_files, segmentation_summary

def compute_pelvis_shifts(mono_data, wham_data, mocap_data, mono_segments, wham_segments, mocap_segments,
                          twocam_data=None, twocam_segments=None):
    """
    Compute the pelvis position shifts between mocap vs mono, mocap vs wham, and mocap vs 2-cameras.
    Uses the shift at the start of first rep as baseline and subtracts it from all shifts.
    Returns absolute values of the differences.
    
    Args:
        mono_data: Numpy structured array with mono data
        wham_data: Numpy structured array with wham data
        mocap_data: Numpy structured array with mocap data
        mono_segments, wham_segments, mocap_segments: Lists of time intervals for each segmentation
        twocam_data: Optional numpy structured array with 2-camera data
        twocam_segments: Optional list of time intervals for 2-camera segmentation
        
    Returns:
        tuple: (mono_shifts, wham_shifts, twocam_shifts) lists of absolute euclidean distances for each repetition,
               adjusted by subtracting initial shift
    """
    pelvis_coords = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    mono_shifts = []
    wham_shifts = []
    twocam_shifts = []
    
    # First, compute the initial shift at the start of first rep
    if len(mono_segments) > 0 and len(wham_segments) > 0 and len(mocap_segments) > 0:
        # Get the start time of first segment
        mocap_start_time = mocap_segments[0][0]  # Start time of first segment
        mono_start_time = mono_segments[0][0]
        wham_start_time = wham_segments[0][0]
        
        # Find the closest time indices
        mocap_start_idx = np.argmin(np.abs(mocap_data['time'] - mocap_start_time))
        mono_start_idx = np.argmin(np.abs(mono_data['time'] - mono_start_time))
        wham_start_idx = np.argmin(np.abs(wham_data['time'] - wham_start_time))
        
        # Get initial pelvis positions
        mocap_init_pos = np.array([mocap_data[coord][mocap_start_idx] for coord in pelvis_coords])
        mono_init_pos = np.array([mono_data[coord][mono_start_idx] for coord in pelvis_coords])
        wham_init_pos = np.array([wham_data[coord][wham_start_idx] for coord in pelvis_coords])
        
        # Compute initial shifts
        mono_init_shift = euclidean(mocap_init_pos, mono_init_pos)
        wham_init_shift = euclidean(mocap_init_pos, wham_init_pos)
        
        # Handle 2-camera initial shift
        twocam_init_shift = 0
        if twocam_data is not None and twocam_segments is not None and len(twocam_segments) > 0:
            twocam_start_time = twocam_segments[0][0]
            twocam_start_idx = np.argmin(np.abs(twocam_data['time'] - twocam_start_time))
            twocam_init_pos = np.array([twocam_data[coord][twocam_start_idx] for coord in pelvis_coords])
            twocam_init_shift = euclidean(mocap_init_pos, twocam_init_pos)
    else:
        return [], [], []
    
    # Now compute shifts at peak of each repetition and subtract initial shift
    num_reps = min(len(mono_segments), len(wham_segments), len(mocap_segments))
    if twocam_data is not None and twocam_segments is not None:
        num_reps = min(num_reps, len(twocam_segments))
    
    for i in range(num_reps):
        # Get the end time of each segment (peak standing position)
        mocap_peak_time = mocap_segments[i][1]  # End time of the segment
        mono_peak_time = mono_segments[i][1]
        wham_peak_time = wham_segments[i][1]
        
        # Find the closest time indices
        mocap_idx = np.argmin(np.abs(mocap_data['time'] - mocap_peak_time))
        mono_idx = np.argmin(np.abs(mono_data['time'] - mono_peak_time))
        wham_idx = np.argmin(np.abs(wham_data['time'] - wham_peak_time))
        
        # Get pelvis positions at peak times
        mocap_pos = np.array([mocap_data[coord][mocap_idx] for coord in pelvis_coords])
        mono_pos = np.array([mono_data[coord][mono_idx] for coord in pelvis_coords])
        wham_pos = np.array([wham_data[coord][wham_idx] for coord in pelvis_coords])
        
        # Compute euclidean distances and subtract initial shift, take absolute value
        mono_shift = abs(euclidean(mocap_pos, mono_pos) - mono_init_shift)
        wham_shift = abs(euclidean(mocap_pos, wham_pos) - wham_init_shift)
        
        mono_shifts.append(mono_shift)
        wham_shifts.append(wham_shift)
        
        # Handle 2-camera shift
        if twocam_data is not None and twocam_segments is not None and i < len(twocam_segments):
            twocam_peak_time = twocam_segments[i][1]
            twocam_idx = np.argmin(np.abs(twocam_data['time'] - twocam_peak_time))
            twocam_pos = np.array([twocam_data[coord][twocam_idx] for coord in pelvis_coords])
            twocam_shift = abs(euclidean(mocap_pos, twocam_pos) - twocam_init_shift)
            twocam_shifts.append(twocam_shift)
    
    return mono_shifts, wham_shifts, twocam_shifts

def plot_pelvis_shifts(mono_shifts, wham_shifts, output_dir, base_name, twocam_shifts=None):
    """
    Create a plot comparing the pelvis position shifts for mono, wham, and 2-cameras relative to mocap.
    
    Args:
        mono_shifts: List of euclidean distances for mono
        wham_shifts: List of euclidean distances for wham
        output_dir: Directory to save the plot
        base_name: Base name for the output file
        twocam_shifts: Optional list of euclidean distances for 2-cameras
    """
    plt.figure(figsize=(10, 6))
    
    x = range(1, len(mono_shifts) + 1)
    
    # Plot shifts
    plt.plot(x, mono_shifts, 'bo-', label='Mono', linewidth=2, markersize=8)
    plt.plot(x, wham_shifts, 'ro-', label='WHAM', linewidth=2, markersize=8)
    if twocam_shifts and len(twocam_shifts) > 0:
        x_twocam = range(1, len(twocam_shifts) + 1)
        plt.plot(x_twocam, twocam_shifts, 'o-', color='orange', label='2-Cameras', linewidth=2, markersize=8)
    
    # Add final value labels at the end of lines
    x_end = len(mono_shifts) + 0.3
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    text_spacing = y_range * 0.02
    
    plt.text(x_end, mono_shifts[-1] + text_spacing, f"{mono_shifts[-1]:.3f}", 
             va='bottom', ha='left', color='blue', fontsize=10)
    plt.text(x_end, wham_shifts[-1], f"{wham_shifts[-1]:.3f}", 
             va='center', ha='left', color='red', fontsize=10)
    if twocam_shifts and len(twocam_shifts) > 0:
        plt.text(x_end, twocam_shifts[-1] - text_spacing, f"{twocam_shifts[-1]:.3f}", 
                 va='top', ha='left', color='orange', fontsize=10)
    
    plt.xlabel('STS Repetition')
    plt.ylabel('Pelvis Position Shift (m) - Absolute Difference with Mocap')
    plt.title('Pelvis Position Shifts Relative to Mocap')
    
    # Set x-axis ticks to whole numbers
    plt.xticks(x)
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{base_name}_pelvis_shifts.svg")
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    return plot_path

def compute_continuous_pelvis_shifts(mono_data, wham_data, mocap_data, twocam_data=None):
    """
    Compute the pelvis position shifts between mocap vs mono, mocap vs wham, and mocap vs 2-cameras
    continuously throughout the entire motion.
    
    Args:
        mono_data: Numpy structured array with mono data
        wham_data: Numpy structured array with wham data
        mocap_data: Numpy structured array with mocap data
        twocam_data: Optional numpy structured array with 2-camera data
        
    Returns:
        tuple: (common_time, mono_shifts, wham_shifts, twocam_shifts, normalized_time) arrays of time points and shifts,
               plus normalized time (0-100% of motion)
    """
    pelvis_coords = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    
    # Get time ranges
    mono_time = mono_data['time']
    wham_time = wham_data['time']
    mocap_time = mocap_data['time']
    
    # Find common time range
    start_time = max(mono_time[0], wham_time[0], mocap_time[0])
    end_time = min(mono_time[-1], wham_time[-1], mocap_time[-1])
    
    # Include 2-camera in time range if available
    if twocam_data is not None:
        twocam_time = twocam_data['time']
        start_time = max(start_time, twocam_time[0])
        end_time = min(end_time, twocam_time[-1])
    
    # Create common time array (use the sampling rate of mocap)
    mocap_dt = np.mean(np.diff(mocap_time))
    common_time = np.arange(start_time, end_time, mocap_dt)
    
    # Create normalized time (0-100%)
    total_duration = end_time - start_time
    normalized_time = np.linspace(0, 100, len(common_time))
    
    # Initialize shift arrays
    mono_shifts = np.zeros_like(common_time)
    wham_shifts = np.zeros_like(common_time)
    twocam_shifts = np.zeros_like(common_time) if twocam_data is not None else None
    
    # Compute initial shift at the first time point
    # Find the closest time indices for the first point
    mocap_start_idx = np.argmin(np.abs(mocap_time - common_time[0]))
    mono_start_idx = np.argmin(np.abs(mono_time - common_time[0]))
    wham_start_idx = np.argmin(np.abs(wham_time - common_time[0]))
    
    # Get initial pelvis positions
    mocap_init_pos = np.array([mocap_data[coord][mocap_start_idx] for coord in pelvis_coords])
    mono_init_pos = np.array([mono_data[coord][mono_start_idx] for coord in pelvis_coords])
    wham_init_pos = np.array([wham_data[coord][wham_start_idx] for coord in pelvis_coords])
    
    # Compute initial shifts
    mono_init_shift = euclidean(mocap_init_pos, mono_init_pos)
    wham_init_shift = euclidean(mocap_init_pos, wham_init_pos)
    
    # Handle 2-camera initial shift
    twocam_init_shift = 0
    if twocam_data is not None:
        twocam_start_idx = np.argmin(np.abs(twocam_time - common_time[0]))
        twocam_init_pos = np.array([twocam_data[coord][twocam_start_idx] for coord in pelvis_coords])
        twocam_init_shift = euclidean(mocap_init_pos, twocam_init_pos)
    
    # Compute shifts at each time point
    for i, t in enumerate(common_time):
        # Find the closest time indices
        mocap_idx = np.argmin(np.abs(mocap_time - t))
        mono_idx = np.argmin(np.abs(mono_time - t))
        wham_idx = np.argmin(np.abs(wham_time - t))
        
        # Get pelvis positions
        mocap_pos = np.array([mocap_data[coord][mocap_idx] for coord in pelvis_coords])
        mono_pos = np.array([mono_data[coord][mono_idx] for coord in pelvis_coords])
        wham_pos = np.array([wham_data[coord][wham_idx] for coord in pelvis_coords])
        
        # Compute euclidean distances and subtract initial shift
        mono_shifts[i] = abs(euclidean(mocap_pos, mono_pos) - mono_init_shift)
        wham_shifts[i] = abs(euclidean(mocap_pos, wham_pos) - wham_init_shift)
        
        # Handle 2-camera
        if twocam_data is not None:
            twocam_idx = np.argmin(np.abs(twocam_time - t))
            twocam_pos = np.array([twocam_data[coord][twocam_idx] for coord in pelvis_coords])
            twocam_shifts[i] = abs(euclidean(mocap_pos, twocam_pos) - twocam_init_shift)
    
    return common_time, mono_shifts, wham_shifts, twocam_shifts, normalized_time

def plot_continuous_pelvis_shifts(common_time, mono_shifts, wham_shifts, twocam_shifts, normalized_time, segments, output_dir, base_name):
    """
    Create a plot showing continuous pelvis position shifts with segment markers.
    
    Args:
        common_time: Array of time points
        mono_shifts: Array of mono shifts
        wham_shifts: Array of wham shifts
        twocam_shifts: Array of 2-camera shifts (can be None)
        normalized_time: Array of normalized time (0-100%)
        segments: List of segment time intervals (for marking STS cycles)
        output_dir: Directory to save the plot
        base_name: Base name for the output file
    """
    # Create two plots: one with absolute time, one with normalized time
    
    # 1. Plot with absolute time
    plt.figure(figsize=(12, 6))
    
    # Plot continuous shifts
    plt.plot(common_time, mono_shifts, 'b-', label='Mono', linewidth=1.5)
    plt.plot(common_time, wham_shifts, 'r-', label='WHAM', linewidth=1.5)
    if twocam_shifts is not None:
        plt.plot(common_time, twocam_shifts, '-', color='orange', label='2-Cameras', linewidth=1.5)
    
    # Add segment markers
    y_min, y_max = plt.ylim()
    height = y_max - y_min
    
    for i, (start, end) in enumerate(segments):
        # Add vertical lines at segment boundaries
        plt.axvline(x=start, color='g', linestyle='--', alpha=0.5)
        plt.axvline(x=end, color='g', linestyle='--', alpha=0.5)
        
        # Add segment number
        mid_time = (start + end) / 2
        plt.text(mid_time, y_max - 0.1*height, f"Rep {i+1}", 
                ha='center', va='top', color='g', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add final value labels at the end of lines
    x_end = common_time[-1]
    y_range = y_max - y_min
    text_spacing = y_range * 0.02
    
    plt.text(x_end, mono_shifts[-1] + text_spacing, f"{mono_shifts[-1]:.3f}", 
             va='bottom', ha='left', color='blue', fontsize=10)
    plt.text(x_end, wham_shifts[-1], f"{wham_shifts[-1]:.3f}", 
             va='center', ha='left', color='red', fontsize=10)
    if twocam_shifts is not None:
        plt.text(x_end, twocam_shifts[-1] - text_spacing, f"{twocam_shifts[-1]:.3f}", 
                 va='top', ha='left', color='orange', fontsize=10)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Pelvis Position Shift (m) - Absolute Difference with Mocap')
    plt.title('Continuous Pelvis Position Shifts Relative to Mocap')
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{base_name}_continuous_shifts.svg")
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    # 2. Plot with normalized time
    plt.figure(figsize=(12, 6))
    
    # Plot continuous shifts
    plt.plot(normalized_time, mono_shifts, 'b-', label='Mono', linewidth=1.5)
    plt.plot(normalized_time, wham_shifts, 'r-', label='WHAM', linewidth=1.5)
    if twocam_shifts is not None:
        plt.plot(normalized_time, twocam_shifts, '-', color='orange', label='2-Cameras', linewidth=1.5)
    
    # Convert segment times to normalized times
    total_duration = common_time[-1] - common_time[0]
    normalized_segments = []
    for start, end in segments:
        norm_start = ((start - common_time[0]) / total_duration) * 100
        norm_end = ((end - common_time[0]) / total_duration) * 100
        normalized_segments.append((norm_start, norm_end))
    
    # Add segment markers
    y_min, y_max = plt.ylim()
    height = y_max - y_min
    
    for i, (start, end) in enumerate(normalized_segments):
        # Add vertical lines at segment boundaries
        plt.axvline(x=start, color='g', linestyle='--', alpha=0.5)
        plt.axvline(x=end, color='g', linestyle='--', alpha=0.5)
        
        # Add segment number
        mid_time = (start + end) / 2
        plt.text(mid_time, y_max - 0.1*height, f"Rep {i+1}", 
                ha='center', va='top', color='g', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add final value labels at the end of lines
    x_end = normalized_time[-1]
    y_range = y_max - y_min
    text_spacing = y_range * 0.02
    
    plt.text(x_end, mono_shifts[-1] + text_spacing, f"{mono_shifts[-1]:.3f}", 
             va='bottom', ha='left', color='blue', fontsize=10)
    plt.text(x_end, wham_shifts[-1], f"{wham_shifts[-1]:.3f}", 
             va='center', ha='left', color='red', fontsize=10)
    if twocam_shifts is not None:
        plt.text(x_end, twocam_shifts[-1] - text_spacing, f"{twocam_shifts[-1]:.3f}", 
                 va='top', ha='left', color='orange', fontsize=10)
    
    plt.xlabel('Normalized Time (%)')
    plt.ylabel('Pelvis Position Shift (m) - Absolute Difference with Mocap')
    plt.title('Continuous Pelvis Position Shifts Relative to Mocap (Normalized Time)')
    
    # Save the plot
    norm_plot_path = os.path.join(output_dir, f"{base_name}_continuous_shifts_normalized.svg")
    plt.savefig(norm_plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    return plot_path, norm_plot_path

def create_comparison_plot(mono_path, wham_path, mocap_path, mono_segments, wham_segments, 
                         mocap_segments, output_dir, base_name, analysis_type='discrete',
                         twocam_path=None, twocam_segments=None):
    """
    Create comparison plots for trajectories and pelvis shifts.
    
    Args:
        mono_path: Path to mono .mot file
        wham_path: Path to wham .mot file
        mocap_path: Path to mocap .mot file (can be None)
        mono_segments, wham_segments, mocap_segments: Lists of time intervals for each segmentation
        output_dir: Directory to save the plot
        base_name: Base name for the output file
        analysis_type: Type of analysis - 'discrete' or 'continuous'
        twocam_path: Optional path to 2-camera .mot file
        twocam_segments: Optional list of time intervals for 2-camera segmentation
    """
    try:
        # Load data
        mono_data = storage_to_numpy(mono_path)
        wham_data = storage_to_numpy(wham_path)
        
        # Extract time and pelvis_ty for trajectory plot
        mono_time = mono_data['time']
        mono_pelvis_ty = mono_data['pelvis_ty']
        
        wham_time = wham_data['time']
        wham_pelvis_ty = wham_data['pelvis_ty']
        
        # Load mocap data if available
        mocap_time = None
        mocap_pelvis_ty = None
        if mocap_path and mocap_segments:
            try:
                mocap_data = storage_to_numpy(mocap_path)
                mocap_time = mocap_data['time']
                mocap_pelvis_ty = mocap_data['pelvis_ty']
            except Exception as e:
                print(f"Error loading mocap data for plot: {str(e)}")
        
        # Load 2-camera data if available
        twocam_data = None
        twocam_time = None
        twocam_pelvis_ty = None
        if twocam_path and os.path.exists(twocam_path):
            try:
                twocam_data = storage_to_numpy(twocam_path)
                twocam_time = twocam_data['time']
                twocam_pelvis_ty = twocam_data['pelvis_ty']
            except Exception as e:
                print(f"Error loading 2-camera data for plot: {str(e)}")
        
        # Create trajectory plot
        plt.figure(figsize=(12, 6))
        
        # Plot mono data
        plt.plot(mono_time, mono_pelvis_ty, 'b-', label='Mono', linewidth=2)
        
        # Plot wham data
        plt.plot(wham_time, wham_pelvis_ty, 'r-', label='WHAM', linewidth=2, alpha=0.7)
        
        # Plot mocap data if available
        if mocap_time is not None and mocap_pelvis_ty is not None:
            plt.plot(mocap_time, mocap_pelvis_ty, 'g-', label='Mocap', linewidth=2, alpha=0.7)
        
        # Plot 2-camera data if available
        if twocam_time is not None and twocam_pelvis_ty is not None:
            plt.plot(twocam_time, twocam_pelvis_ty, '-', color='orange', label='2-Cameras', linewidth=2, alpha=0.7)
        
        # Determine y-axis limits
        y_min = min(np.min(mono_pelvis_ty), np.min(wham_pelvis_ty))
        y_max = max(np.max(mono_pelvis_ty), np.max(wham_pelvis_ty))
        
        if mocap_pelvis_ty is not None:
            y_min = min(y_min, np.min(mocap_pelvis_ty))
            y_max = max(y_max, np.max(mocap_pelvis_ty))
        
        if twocam_pelvis_ty is not None:
            y_min = min(y_min, np.min(twocam_pelvis_ty))
            y_max = max(y_max, np.max(twocam_pelvis_ty))
        
        y_range = y_max - y_min
        y_min -= 0.1 * y_range  # Add 10% padding
        y_max += 0.1 * y_range
        
        # Plot mono segments
        for i, segment in enumerate(mono_segments):
            start_time, end_time = segment
            plt.axvline(x=start_time, color='b', linestyle='--', alpha=0.5)
            plt.axvline(x=end_time, color='b', linestyle='--', alpha=0.5)
            plt.fill_between([start_time, end_time], 
                           y_min, y_min + 0.1 * y_range, 
                           color='blue', alpha=0.2)
            
            # Add text label
            mid_time = (start_time + end_time) / 2
            plt.text(mid_time, y_min + 0.05 * y_range, f"M{i+1}", 
                   color='blue', ha='center')
        
        # Plot wham segments
        for i, segment in enumerate(wham_segments):
            start_time, end_time = segment
            plt.axvline(x=start_time, color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=end_time, color='r', linestyle='--', alpha=0.5)
            plt.fill_between([start_time, end_time], 
                           y_min + 0.1 * y_range, y_min + 0.2 * y_range, 
                           color='red', alpha=0.2)
            
            # Add text label
            mid_time = (start_time + end_time) / 2
            plt.text(mid_time, y_min + 0.15 * y_range, f"W{i+1}", 
                   color='red', ha='center')
        
        # Plot mocap segments if available
        if mocap_segments:
            for i, segment in enumerate(mocap_segments):
                start_time, end_time = segment
                plt.axvline(x=start_time, color='g', linestyle='--', alpha=0.5)
                plt.axvline(x=end_time, color='g', linestyle='--', alpha=0.5)
                plt.fill_between([start_time, end_time], 
                               y_min + 0.2 * y_range, y_min + 0.3 * y_range, 
                               color='green', alpha=0.2)
                
                # Add text label
                mid_time = (start_time + end_time) / 2
                plt.text(mid_time, y_min + 0.25 * y_range, f"C{i+1}", 
                       color='green', ha='center')
        
        # Plot 2-camera segments if available
        if twocam_segments:
            for i, segment in enumerate(twocam_segments):
                start_time, end_time = segment
                plt.axvline(x=start_time, color='orange', linestyle='--', alpha=0.5)
                plt.axvline(x=end_time, color='orange', linestyle='--', alpha=0.5)
                plt.fill_between([start_time, end_time], 
                               y_min + 0.3 * y_range, y_min + 0.4 * y_range, 
                               color='orange', alpha=0.2)
                
                # Add text label
                mid_time = (start_time + end_time) / 2
                plt.text(mid_time, y_min + 0.35 * y_range, f"T{i+1}", 
                       color='orange', ha='center')
        
        # Set y-axis limits
        plt.ylim(y_min, y_max)
        
        # Add labels and title
        plt.xlabel('Time [s]')
        plt.ylabel('Pelvis Height (m)')
        title = 'Comparison of STS Segmentation: Mono vs WHAM'
        if mocap_segments:
            title += ' vs Mocap'
        if twocam_segments:
            title += ' vs 2-Cameras'
        plt.title(title)
        
        # Save the plot
        plot_path = os.path.join(output_dir, f"{base_name}_segmentation_comparison.svg")
        plt.savefig(plot_path, format='svg', bbox_inches='tight')
        plt.close()
        
        # Compute and plot pelvis shifts if mocap data is available
        if mocap_path and mocap_segments:
            try:
                mocap_data = storage_to_numpy(mocap_path)
                
                if analysis_type == 'discrete':
                    # Compute discrete shifts at peaks
                    mono_shifts, wham_shifts, twocam_shifts = compute_pelvis_shifts(
                        mono_data, wham_data, mocap_data,
                        mono_segments, wham_segments, mocap_segments,
                        twocam_data, twocam_segments
                    )
                    
                    # Create discrete shifts plot
                    plot_pelvis_shifts(mono_shifts, wham_shifts, output_dir, base_name, twocam_shifts)
                    
                    # Save shifts to CSV
                    shifts_dict = {
                        'repetition': range(1, len(mono_shifts) + 1),
                        'mono_shift': mono_shifts,
                        'wham_shift': wham_shifts
                    }
                    if twocam_shifts:
                        shifts_dict['twocam_shift'] = twocam_shifts
                    shifts_df = pd.DataFrame(shifts_dict)
                    shifts_csv_path = os.path.join(output_dir, f"{base_name}_pelvis_shifts.csv")
                    shifts_df.to_csv(shifts_csv_path, index=False)
                    
                    # Calculate and save statistics
                    stats = {
                        'mono_mean_shift': np.mean(mono_shifts),
                        'mono_std_shift': np.std(mono_shifts),
                        'wham_mean_shift': np.mean(wham_shifts),
                        'wham_std_shift': np.std(wham_shifts)
                    }
                    if twocam_shifts:
                        stats['twocam_mean_shift'] = np.mean(twocam_shifts)
                        stats['twocam_std_shift'] = np.std(twocam_shifts)
                    
                    stats_path = os.path.join(output_dir, f"{base_name}_shift_stats.json")
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=4)
                
                elif analysis_type == 'continuous':
                    # Compute continuous shifts
                    common_time, mono_shifts, wham_shifts, twocam_shifts, normalized_time = compute_continuous_pelvis_shifts(
                        mono_data, wham_data, mocap_data, twocam_data
                    )
                    
                    # Create continuous shifts plot
                    plot_path, norm_plot_path = plot_continuous_pelvis_shifts(
                        common_time, mono_shifts, wham_shifts, twocam_shifts, normalized_time,
                        mocap_segments, output_dir, base_name
                    )
                    
                    # Save continuous shifts to CSV (downsampled for manageable file size)
                    # Take every 10th point
                    downsample_idx = np.arange(0, len(common_time), 10)
                    shifts_dict = {
                        'time': common_time[downsample_idx],
                        'normalized_time': normalized_time[downsample_idx],
                        'mono_shift': mono_shifts[downsample_idx],
                        'wham_shift': wham_shifts[downsample_idx]
                    }
                    if twocam_shifts is not None:
                        shifts_dict['twocam_shift'] = twocam_shifts[downsample_idx]
                    shifts_df = pd.DataFrame(shifts_dict)
                    shifts_csv_path = os.path.join(output_dir, f"{base_name}_continuous_shifts.csv")
                    shifts_df.to_csv(shifts_csv_path, index=False)
                    
                    # Calculate and save statistics
                    stats = {
                        'mono_mean_shift': float(np.mean(mono_shifts)),
                        'mono_std_shift': float(np.std(mono_shifts)),
                        'wham_mean_shift': float(np.mean(wham_shifts)),
                        'wham_std_shift': float(np.std(wham_shifts))
                    }
                    if twocam_shifts is not None:
                        stats['twocam_mean_shift'] = float(np.mean(twocam_shifts))
                        stats['twocam_std_shift'] = float(np.std(twocam_shifts))
                    
                    stats_path = os.path.join(output_dir, f"{base_name}_continuous_shift_stats.json")
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=4)
                
                else:
                    print(f"Unknown analysis type: {analysis_type}. Using discrete analysis.")
                    # Fall back to discrete analysis
                    mono_shifts, wham_shifts, twocam_shifts = compute_pelvis_shifts(
                        mono_data, wham_data, mocap_data,
                        mono_segments, wham_segments, mocap_segments,
                        twocam_data, twocam_segments
                    )
                    plot_pelvis_shifts(mono_shifts, wham_shifts, output_dir, base_name, twocam_shifts)
                
            except Exception as e:
                print(f"Error computing pelvis shifts: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return plot_path, norm_plot_path
    except Exception as e:
        print(f"Error creating comparison plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def compute_average_shifts(base_folders, output_dir):
    """
    Compute average shifts across all trials by using the normalized time data.
    
    Args:
        base_folders: List of base folder paths
        output_dir: Directory where results are saved
    """
    # Find all continuous shift CSV files
    all_shift_files = []
    for base_folder in base_folders:
        for root, _, files in os.walk(base_folder):
            for file in files:
                if file.endswith('_continuous_shifts.csv'):
                    all_shift_files.append(os.path.join(root, file))
    
    if not all_shift_files:
        print("No continuous shift files found.")
        return
    
    print(f"Found {len(all_shift_files)} continuous shift files for averaging.")
    
    # Create a standard normalized time grid (0-100%)
    standard_time = np.linspace(0, 100, 101)  # 101 points for 0-100%
    
    # Initialize arrays to store interpolated shifts
    all_mono_shifts = []
    all_wham_shifts = []
    all_twocam_shifts = []
    
    # Process each file
    for shift_file in all_shift_files:
        try:
            # Load the data
            df = pd.read_csv(shift_file)
            
            # Interpolate to standard time grid
            mono_interp = np.interp(standard_time, df['normalized_time'], df['mono_shift'])
            wham_interp = np.interp(standard_time, df['normalized_time'], df['wham_shift'])
            
            all_mono_shifts.append(mono_interp)
            all_wham_shifts.append(wham_interp)
            
            # Check if 2-camera data exists
            if 'twocam_shift' in df.columns:
                twocam_interp = np.interp(standard_time, df['normalized_time'], df['twocam_shift'])
                all_twocam_shifts.append(twocam_interp)
            
        except Exception as e:
            print(f"Error processing {shift_file}: {str(e)}")
    
    # Convert to numpy arrays
    all_mono_shifts = np.array(all_mono_shifts)
    all_wham_shifts = np.array(all_wham_shifts)
    has_twocam = len(all_twocam_shifts) > 0
    if has_twocam:
        all_twocam_shifts = np.array(all_twocam_shifts)
    
    # Compute mean and std
    mono_mean = np.mean(all_mono_shifts, axis=0)
    mono_std = np.std(all_mono_shifts, axis=0)
    wham_mean = np.mean(all_wham_shifts, axis=0)
    wham_std = np.std(all_wham_shifts, axis=0)
    if has_twocam:
        twocam_mean = np.mean(all_twocam_shifts, axis=0)
        twocam_std = np.std(all_twocam_shifts, axis=0)
    
    # Create plot with normalized time
    plt.figure(figsize=(12, 6))
    
    # Plot means with shaded std regions
    plt.plot(standard_time, mono_mean, 'b-', label='Mono (Mean)', linewidth=2)
    plt.fill_between(standard_time, mono_mean - mono_std, mono_mean + mono_std, 
                    color='blue', alpha=0.2, label='Mono (±1 SD)')
    
    plt.plot(standard_time, wham_mean, 'r-', label='WHAM (Mean)', linewidth=2)
    plt.fill_between(standard_time, wham_mean - wham_std, wham_mean + wham_std, 
                    color='red', alpha=0.2, label='WHAM (±1 SD)')
    
    if has_twocam:
        plt.plot(standard_time, twocam_mean, '-', color='orange', label='2-Cameras (Mean)', linewidth=2)
        plt.fill_between(standard_time, twocam_mean - twocam_std, twocam_mean + twocam_std, 
                        color='orange', alpha=0.2, label='2-Cameras (±1 SD)')
    
    # Add final value labels at the end of lines
    x_end = standard_time[-1]
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    text_spacing = y_range * 0.02
    
    plt.text(x_end, mono_mean[-1] + text_spacing, f"{mono_mean[-1]:.3f}", 
             va='bottom', ha='left', color='blue', fontsize=10)
    plt.text(x_end, wham_mean[-1], f"{wham_mean[-1]:.3f}", 
             va='center', ha='left', color='red', fontsize=10)
    if has_twocam:
        plt.text(x_end, twocam_mean[-1] - text_spacing, f"{twocam_mean[-1]:.3f}", 
                 va='top', ha='left', color='orange', fontsize=10)
    
    plt.xlabel('Normalized Time (%)')
    plt.ylabel('Pelvis Position Shift (m) - Absolute Difference with Mocap')
    plt.title('Average Pelvis Position Shifts Across All Trials')
    
    # Save the plot
    avg_plot_path = os.path.join(output_dir, "average_pelvis_shifts.svg")
    plt.savefig(avg_plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    # Create a second plot with repetition numbers on x-axis
    # First, find all segment files to determine repetition boundaries
    segment_files = []
    for base_folder in base_folders:
        for root, _, files in os.walk(base_folder):
            for file in files:
                if file.endswith('_segments.csv') and not file.startswith('mocap'):
                    segment_files.append(os.path.join(root, file))
    
    # Determine average number of repetitions
    rep_counts = []
    for file in segment_files:
        try:
            df = pd.read_csv(file)
            rep_counts.append(len(df))
        except:
            pass
    
    if not rep_counts:
        print("Could not determine repetition count from segment files.")
        avg_reps = 5  # Default to 5 repetitions if we can't determine
    else:
        avg_reps = int(np.median(rep_counts))
    
    # Create repetition boundaries (assuming equal distribution)
    rep_boundaries = np.linspace(0, 100, avg_reps + 1)
    
    # Create repetition plot
    plt.figure(figsize=(12, 6))
    
    # Plot means with shaded std regions
    plt.plot(standard_time, mono_mean, 'b-', label='Mono (Mean)', linewidth=2)
    plt.fill_between(standard_time, mono_mean - mono_std, mono_mean + mono_std, 
                    color='blue', alpha=0.2, label='Mono (±1 SD)')
    
    plt.plot(standard_time, wham_mean, 'r-', label='WHAM (Mean)', linewidth=2)
    plt.fill_between(standard_time, wham_mean - wham_std, wham_mean + wham_std, 
                    color='red', alpha=0.2, label='WHAM (±1 SD)')
    
    if has_twocam:
        plt.plot(standard_time, twocam_mean, '-', color='orange', label='2-Cameras (Mean)', linewidth=2)
        plt.fill_between(standard_time, twocam_mean - twocam_std, twocam_mean + twocam_std, 
                        color='orange', alpha=0.2, label='2-Cameras (±1 SD)')
    
    # Add vertical lines for repetition boundaries without labels
    for i, boundary in enumerate(rep_boundaries):
        if i > 0:  # Skip the first boundary (0%)
            plt.axvline(x=boundary, color='g', linestyle='--', alpha=0.5)
    
    # Add repetition numbers at the middle of each segment
    for i in range(avg_reps):
        mid_point = (rep_boundaries[i] + rep_boundaries[i+1]) / 2
        plt.text(mid_point, plt.ylim()[1] * 0.9, f"Repetition {i+1}", 
                ha='center', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add final value labels at the end of lines
    x_end = standard_time[-1]
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    text_spacing = y_range * 0.02
    
    plt.text(x_end, mono_mean[-1] + text_spacing, f"{mono_mean[-1]:.3f}", 
             va='bottom', ha='left', color='blue', fontsize=10)
    plt.text(x_end, wham_mean[-1], f"{wham_mean[-1]:.3f}", 
             va='center', ha='left', color='red', fontsize=10)
    if has_twocam:
        plt.text(x_end, twocam_mean[-1] - text_spacing, f"{twocam_mean[-1]:.3f}", 
                 va='top', ha='left', color='orange', fontsize=10)
    
    plt.xlabel('Normalized Time (%) with Repetition Markers')
    plt.ylabel('Pelvis Position Shift (m) - Absolute Difference with Mocap')
    plt.title('Average Pelvis Position Shifts by Repetition')
    
    # Save the plot
    rep_plot_path = os.path.join(output_dir, "average_pelvis_shifts_by_repetition.svg")
    plt.savefig(rep_plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    # Save the data
    avg_data_dict = {
        'normalized_time': standard_time,
        'mono_mean': mono_mean,
        'mono_std': mono_std,
        'wham_mean': wham_mean,
        'wham_std': wham_std
    }
    if has_twocam:
        avg_data_dict['twocam_mean'] = twocam_mean
        avg_data_dict['twocam_std'] = twocam_std
    avg_data = pd.DataFrame(avg_data_dict)
    
    avg_data_path = os.path.join(output_dir, "average_pelvis_shifts.csv")
    avg_data.to_csv(avg_data_path, index=False)
    
    print(f"Average shifts computed and saved to {avg_plot_path}, {rep_plot_path}, and {avg_data_path}")
    
    # Calculate overall statistics
    overall_stats = {
        'mono_overall_mean': float(np.mean(mono_mean)),
        'mono_overall_std': float(np.mean(mono_std)),
        'wham_overall_mean': float(np.mean(wham_mean)),
        'wham_overall_std': float(np.mean(wham_std))
    }
    if has_twocam:
        overall_stats['twocam_overall_mean'] = float(np.mean(twocam_mean))
        overall_stats['twocam_overall_std'] = float(np.mean(twocam_std))
    
    stats_path = os.path.join(output_dir, "overall_shift_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(overall_stats, f, indent=4)
    
    return avg_data

def compute_average_discrete_shifts(base_folders, output_dir):
    """
    Compute average discrete shifts across all trials.
    
    Args:
        base_folders: List of base folder paths
        output_dir: Directory where results are saved
    """
    # Find all discrete shift CSV files
    all_shift_files = []
    for base_folder in base_folders:
        for root, _, files in os.walk(base_folder):
            for file in files:
                if file.endswith('_pelvis_shifts.csv') and not file.endswith('_continuous_shifts.csv'):
                    all_shift_files.append(os.path.join(root, file))
    
    if not all_shift_files:
        print("No discrete shift files found.")
        return
    
    print(f"Found {len(all_shift_files)} discrete shift files for averaging.")
    
    # Determine the maximum number of repetitions
    max_reps = 0
    for shift_file in all_shift_files:
        try:
            df = pd.read_csv(shift_file)
            max_reps = max(max_reps, len(df))
        except Exception as e:
            print(f"Error reading {shift_file}: {str(e)}")
    
    if max_reps == 0:
        print("No valid repetition data found.")
        return
    
    # Initialize arrays to store shifts for each repetition
    all_mono_shifts = [[] for _ in range(max_reps)]
    all_wham_shifts = [[] for _ in range(max_reps)]
    all_twocam_shifts = [[] for _ in range(max_reps)]
    has_twocam = False
    
    # Process each file
    for shift_file in all_shift_files:
        try:
            # Load the data
            df = pd.read_csv(shift_file)
            
            # Check if 2-camera data exists
            file_has_twocam = 'twocam_shift' in df.columns
            if file_has_twocam:
                has_twocam = True
            
            # Add shifts to the appropriate repetition list
            for i, row in df.iterrows():
                rep_idx = int(row['repetition']) - 1
                if rep_idx < max_reps:
                    all_mono_shifts[rep_idx].append(row['mono_shift'])
                    all_wham_shifts[rep_idx].append(row['wham_shift'])
                    if file_has_twocam:
                        all_twocam_shifts[rep_idx].append(row['twocam_shift'])
            
        except Exception as e:
            print(f"Error processing {shift_file}: {str(e)}")
    
    # Compute mean and std for each repetition
    mono_means = []
    mono_stds = []
    wham_means = []
    wham_stds = []
    twocam_means = []
    twocam_stds = []
    
    for rep_idx in range(max_reps):
        if all_mono_shifts[rep_idx] and all_wham_shifts[rep_idx]:
            mono_means.append(np.mean(all_mono_shifts[rep_idx]))
            mono_stds.append(np.std(all_mono_shifts[rep_idx]))
            wham_means.append(np.mean(all_wham_shifts[rep_idx]))
            wham_stds.append(np.std(all_wham_shifts[rep_idx]))
        else:
            # If no data for this repetition, use NaN
            mono_means.append(np.nan)
            mono_stds.append(np.nan)
            wham_means.append(np.nan)
            wham_stds.append(np.nan)
        
        # Handle 2-camera data
        if has_twocam and all_twocam_shifts[rep_idx]:
            twocam_means.append(np.mean(all_twocam_shifts[rep_idx]))
            twocam_stds.append(np.std(all_twocam_shifts[rep_idx]))
        elif has_twocam:
            twocam_means.append(np.nan)
            twocam_stds.append(np.nan)
    
    # Create x-axis (repetition numbers)
    x = np.arange(1, max_reps + 1)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot means with error bars
    plt.errorbar(x, mono_means, yerr=mono_stds, fmt='bo-', capsize=5, 
                label='Mono (Mean ± SD)', linewidth=2, markersize=8)
    
    plt.errorbar(x, wham_means, yerr=wham_stds, fmt='ro-', capsize=5, 
                label='WHAM (Mean ± SD)', linewidth=2, markersize=8)
    
    if has_twocam:
        plt.errorbar(x, twocam_means, yerr=twocam_stds, fmt='o-', color='orange', capsize=5, 
                    label='2-Cameras (Mean ± SD)', linewidth=2, markersize=8)
    
    # Add final value labels at the end of the lines
    x_end = max_reps + 0.3
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    text_spacing = y_range * 0.02
    
    if not np.isnan(mono_means[-1]):
        plt.text(x_end, mono_means[-1] + text_spacing, f"{mono_means[-1]:.3f}", 
                 va='bottom', ha='left', color='blue', fontsize=10)
    if not np.isnan(wham_means[-1]):
        plt.text(x_end, wham_means[-1], f"{wham_means[-1]:.3f}", 
                 va='center', ha='left', color='red', fontsize=10)
    if has_twocam and not np.isnan(twocam_means[-1]):
        plt.text(x_end, twocam_means[-1] - text_spacing, f"{twocam_means[-1]:.3f}", 
                 va='top', ha='left', color='orange', fontsize=10)
    
    plt.xlabel('STS Repetition')
    plt.ylabel('Pelvis Position Shift (m) - Absolute Difference with Mocap')
    plt.title('Average Pelvis Position Shifts Across All Trials')
    
    # Set x-axis ticks to whole numbers
    plt.xticks(x)
    
    # Save the plot
    avg_plot_path = os.path.join(output_dir, "average_discrete_pelvis_shifts.svg")
    plt.savefig(avg_plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    # Save the data
    avg_data_dict = {
        'repetition': x,
        'mono_mean': mono_means,
        'mono_std': mono_stds,
        'wham_mean': wham_means,
        'wham_std': wham_stds
    }
    if has_twocam:
        avg_data_dict['twocam_mean'] = twocam_means
        avg_data_dict['twocam_std'] = twocam_stds
    avg_data = pd.DataFrame(avg_data_dict)
    
    avg_data_path = os.path.join(output_dir, "average_discrete_pelvis_shifts.csv")
    avg_data.to_csv(avg_data_path, index=False)
    
    print(f"Average discrete shifts computed and saved to {avg_plot_path} and {avg_data_path}")
    
    # Calculate overall statistics
    valid_mono_means = [m for m in mono_means if not np.isnan(m)]
    valid_mono_stds = [s for s in mono_stds if not np.isnan(s)]
    valid_wham_means = [m for m in wham_means if not np.isnan(m)]
    valid_wham_stds = [s for s in wham_stds if not np.isnan(s)]
    
    overall_stats = {
        'mono_overall_mean': float(np.mean(valid_mono_means)) if valid_mono_means else 0,
        'mono_overall_std': float(np.mean(valid_mono_stds)) if valid_mono_stds else 0,
        'wham_overall_mean': float(np.mean(valid_wham_means)) if valid_wham_means else 0,
        'wham_overall_std': float(np.mean(valid_wham_stds)) if valid_wham_stds else 0
    }
    
    if has_twocam:
        valid_twocam_means = [t for t in twocam_means if not np.isnan(t)]
        valid_twocam_stds = [s for s in twocam_stds if not np.isnan(s)]
        overall_stats['twocam_overall_mean'] = float(np.mean(valid_twocam_means)) if valid_twocam_means else 0
        overall_stats['twocam_overall_std'] = float(np.mean(valid_twocam_stds)) if valid_twocam_stds else 0
    
    stats_path = os.path.join(output_dir, "overall_discrete_shift_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(overall_stats, f, indent=4)
    
    return avg_data

def main():
    # Specify the base folders to search
    base_folders = [
        "../output/nas/case_001_STS",
    ]
    
    # Define output directory
    output_dir = os.path.join(os.getcwd(), "sts_analysis_results")
    
    # Specify analysis type ('discrete' or 'continuous')
    analysis_type = 'continuous'  # Change to 'continuous' for continuous analysis
    
    # Analyze STS trials
    processed_files, segmentation_summary = analyze_sts_trials(
        base_folders, output_dir, visualize=False, analysis_type=analysis_type)
    
    print(f"STS analysis complete. Found {len(processed_files)} trials.")
    
    # Compute average shifts across all trials
    if analysis_type == 'continuous':
        compute_average_shifts(base_folders, output_dir)
    else:  # discrete analysis
        compute_average_discrete_shifts(base_folders, output_dir)

if __name__ == "__main__":
    main()