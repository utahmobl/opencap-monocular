import json
import os


def get_fps_from_json(json_data):
    """
    Compute fps from time values in json data

    Args:
        json_data (dict): Loaded JSON data containing 'time' field

    Returns:
        float: Computed frames per second
    """
    time_values = json_data["time"]
    if len(time_values) < 2:
        return 100  # default fallback

    # Compute average time step
    time_step = (time_values[-1] - time_values[0]) / (len(time_values) - 1)
    fps = round(1 / time_step)
    return fps


def automate_recording(json_paths, output_video_path, wait_time=5, num_loops=1):
    """
    Generate video from JSON files using opencap_visualizer Python API.

    Args:
        json_paths (list): List of paths to JSON files
        output_video_path (str): Path where the video should be saved
        wait_time (int): Time to wait for animation to load in seconds (unused with Python API)
        num_loops (int): Number of animation loops to record
    """
    # Import the opencap_visualizer package
    try:
        import opencap_visualizer as ocv
    except ImportError:
        print("Error: opencap-visualizer package not found. Please install it with:")
        print("pip install opencap-visualizer")
        raise ImportError("opencap-visualizer package is required")

    # Verify input files exist
    for i, json_path in enumerate(json_paths):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file {i+1} not found: {json_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load and parse the first JSON file to get fps info
    try:
        with open(json_paths[0], "r") as f:
            json1_data = json.load(f)

        fps = get_fps_from_json(json1_data)
        print(f"Detected frame rate: {fps} fps")
    except Exception as e:
        print(f"Warning: Could not parse JSON file for FPS detection: {e}")

    try:
        # Determine input format - single file or multiple files
        input_data = json_paths[0] if len(json_paths) == 1 else json_paths

        print(f"Generating video from {len(json_paths)} JSON file(s)...")
        print(f"Output: {output_video_path}")
        print(f"Loops: {num_loops}")

        # Generate different colors for multiple subjects
        colors = None
        if len(json_paths) > 1:
            color_options = [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "yellow",
                "cyan",
                "magenta",
            ]
            colors = color_options[: len(json_paths)]

        # Call the opencap_visualizer API
        success = ocv.create_video(
            input_data, output_video_path, loops=num_loops, colors=colors, verbose=False
        )

        if success:
            print(f"Video generated successfully: {output_video_path}")
        else:
            print("Failed to generate video")
            raise RuntimeError("Video generation failed")

    except Exception as e:
        print(f"Error generating video with opencap_visualizer: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automate animation recording")
    parser.add_argument(
        "json_files", nargs="+", help="Paths to JSON files (2 or 3 files)"
    )
    parser.add_argument("output", help="Path for output video file")
    parser.add_argument(
        "--wait", type=int, default=5, help="Wait time for loading in seconds"
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Number of animation loops to record with different camera angles",
    )

    args = parser.parse_args()

    if not (2 <= len(args.json_files) <= 3):
        parser.error("You must provide 2 or 3 JSON files")

    automate_recording(args.json_files, args.output, args.wait, args.loops)
