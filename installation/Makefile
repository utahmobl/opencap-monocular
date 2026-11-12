.PHONY: help setup setup-intrinsics update-intrinsics check-intrinsics clean-intrinsics install test

help:
	@echo "Available targets:"
	@echo "  setup             - Complete setup including camera intrinsics"
	@echo "  setup-intrinsics  - Sync camera intrinsics (download if missing/outdated)"
	@echo "  update-intrinsics - Force re-download latest intrinsics from opencap-org/opencap-core"
	@echo "  check-intrinsics  - Exit 0 if local matches remote CameraIntrinsics tip"
	@echo "  clean-intrinsics  - Remove camera intrinsics data"
	@echo "  install           - Install Python dependencies"
	@echo "  test              - Run tests (if available)"
	@echo "  help              - Show this help message"

setup: install setup-intrinsics
	@echo "Setup complete!"

_run-intrinsics-script:
	@if command -v python3 >/dev/null 2>&1; then \
		python3 setup_camera_intrinsics.py $(ARGS); \
	else \
		python setup_camera_intrinsics.py $(ARGS); \
	fi

setup-intrinsics:
	@echo "Syncing camera intrinsics (updates when remote CameraIntrinsics changes)..."
	@$(MAKE) _run-intrinsics-script ARGS=""

update-intrinsics:
	@echo "Updating camera intrinsics from opencap-org/opencap-core..."
	@$(MAKE) _run-intrinsics-script ARGS="--update"

check-intrinsics:
	@$(MAKE) _run-intrinsics-script ARGS="--check"

clean-intrinsics:
	@echo "Removing camera intrinsics data..."
	@rm -rf ../camera_intrinsics/
	@rm -rf ../temp_opencap_core/
	@echo "Camera intrinsics data removed."

install:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt

test:
	@echo "Running tests..."
	@echo "No tests configured yet." 