#!/bin/bash

# Setup script to download CameraIntrinsics from opencap-core repository
# This script ensures the camera intrinsics data is available for OpenCap processing

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CAMERA_INTRINSICS_DIR="$PROJECT_ROOT/camera_intrinsics"
TEMP_DIR="$PROJECT_ROOT/temp_opencap_core"

echo "Setting up camera intrinsics data..."

# Check if camera_intrinsics directory already exists and has content
if [ -d "$CAMERA_INTRINSICS_DIR" ] && [ "$(ls -A "$CAMERA_INTRINSICS_DIR" 2>/dev/null)" ]; then
    echo "Camera intrinsics directory already exists and contains data. Skipping download."
    echo "To force re-download, delete the camera_intrinsics directory and run this script again."
    exit 0
fi

# Remove existing empty directory if it exists
if [ -d "$CAMERA_INTRINSICS_DIR" ]; then
    echo "Removing empty camera_intrinsics directory..."
    rm -rf "$CAMERA_INTRINSICS_DIR"
fi

# Clone the opencap-core repository with sparse checkout
echo "Downloading camera intrinsics from opencap-core repository..."
cd "$PROJECT_ROOT"
git clone --depth 1 --filter=blob:none --sparse https://github.com/opencap-org/opencap-core.git "$TEMP_DIR"

# Navigate to temp directory and set up sparse checkout
cd "$TEMP_DIR"
git sparse-checkout set CameraIntrinsics

# Copy the CameraIntrinsics folder to the target location
echo "Copying camera intrinsics data..."
cp -r CameraIntrinsics "$CAMERA_INTRINSICS_DIR"

# Clean up temporary directory
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"

echo "Camera intrinsics setup complete!"
echo "Data available in: $CAMERA_INTRINSICS_DIR"

# Count the number of device models available
DEVICE_COUNT=$(find "$CAMERA_INTRINSICS_DIR" -maxdepth 1 -type d | grep -E "(iPhone|iPad|iPod)" | wc -l)
echo "Available device models: $DEVICE_COUNT" 