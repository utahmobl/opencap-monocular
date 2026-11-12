# OpenCap Viewer Python Automation

A Python automation tool for recording OpenCap motion capture animations using the OpenCap web viewer.

## Features

- Automated recording of OpenCap motion capture comparisons
- Multiple camera angle recordings
- Customizable wait times and recording loops
- Automatic file handling and cleanup

## Prerequisites

- Python 3.7+
- Google Chrome browser
- Internet connection (to access the OpenCap viewer website)

## Installation

1. Create and activate a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Use the Python script to automate the recording process:

```bash
python automation.py path/to/first.json path/to/second.json output_video.webm
```

Optional arguments:
- `--wait`: Adjust loading wait time (default: 5 seconds)
- `--loops`: Number of camera angle changes (default: 3)

Example:
```bash
python automation.py test1.json test2.json comparison.webm --wait 10 --loops 4
```

### Camera Angles

The automated recording cycles through these views:
1. Front view (0°)
2. Side view (90°)
3. Back view (180°)
4. High angle diagonal view (45°)

## File Format

The script expects OpenCap JSON files with the following structure:
```json
{
  "time": [...],
  "bodies": {
    "body_name": {
      "translation": [...],
      "rotation": [...],
      "attachedGeometries": [...]
    }
  }
}
```

## Troubleshooting

1. If the video doesn't download:
   - Check Chrome's download permissions
   - Ensure the output directory is writable
   - Verify your internet connection

2. If models don't appear:
   - Verify JSON file format
   - Check that the files are valid OpenCap JSON files
   - Increase the wait time using `--wait`

3. If camera controls don't work:
   - Ensure the browser window is focused
   - Check that the website has loaded properly

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
