#!/usr/bin/env python3
"""
Script to run the OpenCap pipeline with resource monitoring.

This script runs the flow function with resource monitoring enabled,
tracking GPU and CPU usage throughout the pipeline execution.
"""

import os
import sys
import argparse
from loguru import logger

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.flow import flow


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenCap pipeline with resource monitoring"
    )

    parser.add_argument(
        "--subject", default="subject4", help="Subject ID (default: subject4)"
    )
    parser.add_argument(
        "--session", default="Session0", help="Session ID (default: Session0)"
    )
    parser.add_argument("--cam", default="Cam3", help="Camera ID (default: Cam3)")
    parser.add_argument(
        "--video", default="squats1", help="Video name (default: squats1)"
    )
    parser.add_argument(
        "--video-name",
        default="squats1.avi",
        help="Video filename (default: squats1.avi)",
    )
    parser.add_argument(
        "--case-num",
        default="case_scaling_fixed",
        help="Case number (default: case_scaling_fixed)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=True,
        help="Rerun pipeline (default: True)",
    )
    parser.add_argument(
        "--no-monitoring", action="store_true", help="Disable resource monitoring"
    )

    args = parser.parse_args()

    # Configure logging
    logger.add(sys.stderr, level="INFO")

    print("=" * 80)
    print("OPENCAP PIPELINE WITH RESOURCE MONITORING")
    print("=" * 80)
    print(f"Subject: {args.subject}")
    print(f"Session: {args.session}")
    print(f"Camera: {args.cam}")
    print(f"Video: {args.video}")
    print(f"Video file: {args.video_name}")
    print(f"Case: {args.case_num}")
    print(f"Rerun: {args.rerun}")
    print(f"Monitoring: {not args.no_monitoring}")
    print("=" * 80)

    try:
        # Run the flow with monitoring
        flow(
            subject=args.subject,
            session=args.session,
            cam=args.cam,
            video=args.video,
            video_name=args.video_name,
            case_num=args.case_num,
            rerun=args.rerun,
            enable_monitoring=not args.no_monitoring,
        )

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Resource monitoring results have been saved to the output directory.")
        print("Check the 'monitoring' folder in your case output directory for:")
        print("- resource_usage_data.json: Raw monitoring data")
        print("- resource_usage_plot.png: Resource usage plots")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print("\n" + "=" * 80)
        print("PIPELINE FAILED!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
