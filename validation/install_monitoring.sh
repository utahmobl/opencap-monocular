#!/bin/bash

# Installation script for OpenCap resource monitoring dependencies

echo "=========================================="
echo "Installing OpenCap Resource Monitoring"
echo "=========================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating your virtual environment first"
    echo "   Example: source venv/bin/activate"
    echo ""
fi

# Install required packages
echo "Installing required packages..."
pip install -r requirements_monitoring.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Required packages installed successfully"
else
    echo "❌ Failed to install required packages"
    exit 1
fi

# Test if pynvml is available
echo "Testing GPU monitoring capabilities..."
python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')

try:
    import pynvml
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    print(f'NVML available: True (Found {gpu_count} GPU(s))')
except ImportError:
    print('NVML available: False (pynvml not installed)')
except Exception as e:
    print(f'NVML available: False (Error: {e})')
"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test the monitoring system:"
echo "   python validation/test_monitoring.py"
echo ""
echo "2. Run the pipeline with monitoring:"
echo "   python validation/run_monitored_flow.py"
echo ""
echo "3. Check the README for more details:"
echo "   cat validation/README_monitoring.md"
echo "" 