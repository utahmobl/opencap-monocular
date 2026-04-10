import os
import cv2


def convert_to_avi(inputPath, outputPath=None, frameRate=None, quality=0, rotation=None):
    """
    Convert video format and optionally rotate frames
    
    Args:
        inputPath:  Path to input video
        outputPath: Path to output video (if none, will be the same as inputPath but with .avi extension)
        frameRate: Optional frame rate limit (e.g., 60)
        quality: Quality setting for output (0=best, default)
        rotation: Rotation angle in degrees (0, 90, 180, 270). If None, uses ffmpeg conversion.
                  If specified, uses OpenCV to rotate frames during conversion.
    """
    if outputPath is None:
        outputPath = inputPath.replace(".mov", ".avi").replace(".mp4", ".avi")
    
    # If rotation is specified and needs frame rotation, use OpenCV to rotate frames (like old working version)
    # Old version: should_rotate=True for 90/270 meant frames needed rotation to upright
    # New version: rotation=90/270 means frames are rotated, need to rotate them to upright
    # Rotation 0/180: frames are already in correct orientation, no rotation needed
    if rotation is not None and rotation in [90, 270]:
        return _convert_with_rotation(inputPath, outputPath, rotation, frameRate, quality)
    else:
        # Use ffmpeg for simple conversion (no rotation needed for 0, 180, or None)
        cmd_fr = '' if frameRate is None else f' -r {frameRate} '
        CMD = f"ffmpeg -loglevel error -y -i {inputPath}{cmd_fr} -q:v {quality} {outputPath}"
        
        if not os.path.exists(outputPath):
            os.system(CMD)
        
        return outputPath


def _convert_with_rotation(input_path, output_path, rotation, frame_rate=None, quality=0):
    """
    Convert video using OpenCV and rotate frames
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        rotation: Rotation angle in degrees (90, 180, 270)
        frame_rate: Optional frame rate limit
        quality: Quality setting (0=best)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate is not None:
        fps = min(fps, frame_rate)
    
    # Determine output dimensions based on rotation
    if rotation == 90:
        output_width = height
        output_height = width
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotation == 270:
        output_width = height
        output_height = width
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    elif rotation == 180:
        output_width = width
        output_height = height
        rotate_code = cv2.ROTATE_180
    else:
        output_width = width
        output_height = height
        rotate_code = None
    
    # Use XVID codec (same as old version)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video file: {output_path}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if rotate_code is not None:
            rotated_frame = cv2.rotate(frame, rotate_code)
            out.write(rotated_frame)
        else:
            out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return output_path

if __name__ == "__main__":
    original_video_path = "../opencap/Data/cf9c3cfb-b532-424e-9f7e-346586d920c4/Videos/Cam0/InputMedia/tug_sideView/d01c11f9-45cd-4159-8fcd-7c611f2f620f.mov"
    output_video_path = "test_output.avi"
    convert_to_avi(original_video_path, outputPath=output_video_path)
