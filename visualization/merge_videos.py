# walk through all folders in the output dir, find all files with .webm extension, and merge them into a single video.
# write the video path on each frame of the merged video to be able to identify the video that each frame belongs to.
# save the merged video to the output dir.

import os
import cv2
import numpy as np
import datetime
import re


def merge_videos(
    output_dir,
    movements=None,
    modified_after=None,
    base_folders=None,
    camera_numbers=None,
):

    # Find all .webm files recursively
    video_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("viewer_mono.webm") or file.endswith("recording.webm"):
                file_path = os.path.join(root, file)

                # Apply base_folders filter if specified
                if base_folders is not None:
                    # Check if the file is in one of the specified base folders
                    if not any(
                        base_folder in file_path for base_folder in base_folders
                    ):
                        continue

                # Apply camera_numbers filter if specified
                if camera_numbers is not None:
                    # Extract camera number from path
                    camera_match = re.search(r"Cam(\d+)", file_path)
                    if camera_match:
                        camera_num = int(camera_match.group(1))
                        if camera_num not in camera_numbers:
                            continue
                    else:
                        # Skip if we can't determine the camera number
                        continue

                # Apply modified_after filter if specified
                if modified_after is not None:
                    # Convert datetime to timestamp if needed
                    if isinstance(modified_after, datetime.datetime):
                        modified_after_ts = modified_after.timestamp()
                    else:
                        modified_after_ts = modified_after

                    # Check modification time
                    mod_time = os.path.getmtime(file_path)
                    if mod_time < modified_after_ts:
                        continue

                video_files.append(file_path)

    if not video_files:
        print("No .webm files found")
        return

    print(f"Found {len(video_files)} videos")

    # print(video_files)

    if movements is not None:
        # filter the video files to only include videos which contain one of the movements names. lower case
        video_files = [
            video_file
            for video_file in video_files
            if any(movement.lower() in video_file.lower() for movement in movements)
        ]

    if len(video_files) == 0:
        print(f"No videos found for movements {movements}")
        return

    print(f"Found {len(video_files)} videos for movements {movements}")

    # Get video properties from first file
    cap = cv2.VideoCapture(video_files[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 60
    print(f"FPS: {fps}")
    cap.release()

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # add the movements to the output path
    output_path = os.path.join(output_dir, "_".join(movements) + "_merged_videos.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each video
    for video_path in video_files:
        print(
            f"Processing video {video_files.index(video_path) + 1} of {len(video_files)}"
        )
        cap = cv2.VideoCapture(video_path)

        # Get relative path for display
        rel_path = os.path.relpath(video_path, output_dir)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Add video path text to frame
            cv2.putText(
                frame,
                rel_path,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            out.write(frame)

        cap.release()

    out.release()
    print(f"Merged video saved to: {output_path}")


# def delete all videos ending with .webm in the output dir
def delete_videos(output_dir):
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".webm"):
                os.remove(os.path.join(root, file))
                print(f"Deleted {os.path.join(root, file)}")


if __name__ == "__main__":
    # Get repo root directory
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(repo_path, "output")

    # Define specific folders and camera numbers to filter by
    base_folders = [
        "output/case_001_walking",
        "output/case_001_STS",
        "output/case_001_squats",
    ]
    camera_numbers = [3]

    movements = ["squat", "walking", "sts"]  # ['squat', 'walking', 'sts']

    # Example: Only merge videos modified after today at 9:30 AM
    # time_th = datetime.datetime.combine(datetime.date.today(), datetime.time(11, 30))
    # merge_videos(output_dir, movements=movements, modified_after=time_th)

    # With folder and camera filters
    merge_videos(
        output_dir,
        movements=movements,
        base_folders=base_folders,
        camera_numbers=camera_numbers,
    )

    # Without any filters (original behavior)
    # merge_videos(output_dir, movements=movements)
