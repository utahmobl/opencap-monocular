import sys
import matplotlib.pyplot as plt
import numpy as np
from utils.utils_trc import TRCFile


def parse_trc(file_path):
    """
    Parse a .trc file using the TRCFile class and return time, right ankle Y, left ankle Y arrays.
    """
    try:
        # Use the proper TRCFile parser
        trc = TRCFile(file_path)
        
        print(f"\nDebugging {file_path}:")
        print(f"Found {len(trc.marker_names)} markers:")
        print(f"Units: {trc.units}")
        print(f"Data rate: {trc.data_rate} Hz")
        print(f"Number of frames: {trc.num_frames}")
        
        # Find ankle markers
        right_ankle_name = None
        left_ankle_name = None
        
        for name in trc.marker_names:
            name_lower = name.lower()
            if 'r_ankle' in name_lower or 'rankle' in name_lower or 'r.ankle' in name_lower:
                right_ankle_name = name
                print(f"Found right ankle: {name}")
            elif 'l_ankle' in name_lower or 'lankle' in name_lower or 'l.ankle' in name_lower:
                left_ankle_name = name
                print(f"Found left ankle: {name}")
        
        if right_ankle_name is None or left_ankle_name is None:
            print(f"Could not find ankle markers. Available markers with 'ankle':")
            for name in trc.marker_names:
                if 'ankle' in name.lower():
                    print(f"  {name}")
            return [], [], []
        
        # Extract marker data using the TRCFile methods
        right_ankle_data = trc.marker(right_ankle_name)
        left_ankle_data = trc.marker(left_ankle_name)
        
        # Y coordinate is the second column (index 1)
        right_ankle_y = right_ankle_data[:, 1]
        left_ankle_y = left_ankle_data[:, 1]
        
        # Print some sample values to understand the scale
        print(f"Sample right ankle Y values (first 5 frames): {right_ankle_y[:5]}")
        print(f"Sample left ankle Y values (first 5 frames): {left_ankle_y[:5]}")
        
        print(f"Parsed {len(trc.time)} data points")
        return trc.time, right_ankle_y, left_ankle_y
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], [], []


def plot_ankle_height(file_paths):
    plt.figure(figsize=(12, 6))
    
    all_data = []
    for file_path in file_paths:
        try:
            time, ry, ly = parse_trc(file_path)
            
            # Check if we have valid data
            if len(time) == 0 or len(ry) == 0 or len(ly) == 0:
                print(f"Warning: No valid data found in {file_path}")
                continue
                
            all_data.extend([ry, ly])
            
            # Print data ranges for debugging
            print(f"\nFile: {file_path}")
            print(f"Time range: {time.min():.3f} to {time.max():.3f} seconds")
            print(f"Right ankle Y range: {ry.min():.3f} to {ry.max():.3f} meters")
            print(f"Left ankle Y range: {ly.min():.3f} to {ly.max():.3f} meters")
            
            # Create a shorter label for the plot
            file_label = file_path.split('/')[-1]  # Just the filename
            plt.plot(time, ry, label=f'Right Ankle - {file_label}', linewidth=1.5)
            plt.plot(time, ly, label=f'Left Ankle - {file_label}', linewidth=1.5)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_data:
        print("No valid data found in any files!")
        plt.close()
        return
    
    plt.xlabel('Time (s)')
    plt.ylabel('Ankle Height (m)')
    plt.title('Ankle Height Over Time (Y Coordinate)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Auto-scale y-axis to fit data
    if all_data:
        all_data = np.concatenate(all_data)
        y_min, y_max = all_data.min(), all_data.max()
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    default_files = [
        '/home/selim/Downloads/2nd/opencap.trc',
        '/home/selim/opencap-mono/results/second_2/vid2/MarkerData/vid2_5/vid2_5.trc'
    ]
    if len(sys.argv) < 2:
        print('No input files provided. Using default files:')
        for f in default_files:
            print(f'  {f}')
        plot_ankle_height(default_files)
    else:
        plot_ankle_height(sys.argv[1:])
