import os
import time
import threading
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import deque
import json
from loguru import logger


class ResourceMonitor:
    """
    Monitor GPU and CPU usage over time and generate plots.
    """

    def __init__(self, max_samples=1000, sampling_interval=1.0):
        """
        Initialize the resource monitor.

        Args:
            max_samples (int): Maximum number of samples to store in memory
            sampling_interval (float): Sampling interval in seconds
        """
        self.max_samples = max_samples
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread = None

        # Data storage
        self.timestamps = deque(maxlen=max_samples)
        self.cpu_percent = deque(maxlen=max_samples)
        self.memory_percent = deque(maxlen=max_samples)
        self.memory_gb = deque(maxlen=max_samples)
        self.gpu_memory_allocated = deque(maxlen=max_samples)
        self.gpu_memory_reserved = deque(maxlen=max_samples)
        self.gpu_utilization = deque(maxlen=max_samples)

        # Process-specific monitoring
        self.process = psutil.Process()

        # GPU monitoring setup
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml_available = True
                self.device_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"NVML initialized. Found {self.device_count} GPU(s)")
            except ImportError:
                logger.warning(
                    "pynvml not available. GPU utilization monitoring disabled."
                )
                self.nvml_available = False
        else:
            self.nvml_available = False

    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._collect_sample()
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)

    def _collect_sample(self):
        """Collect a single sample of resource usage."""
        timestamp = datetime.now()

        # CPU and memory
        cpu_pct = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # Process-specific memory
        process_memory = self.process.memory_info()

        # GPU memory (PyTorch)
        gpu_allocated = 0
        gpu_reserved = 0
        if self.gpu_available:
            try:
                gpu_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
            except Exception as e:
                logger.debug(f"Error getting GPU memory: {e}")

        # GPU utilization (NVML)
        gpu_util = 0
        if self.nvml_available:
            try:
                import pynvml

                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Monitor first GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception as e:
                logger.debug(f"Error getting GPU utilization: {e}")

        # Store data
        self.timestamps.append(timestamp)
        self.cpu_percent.append(cpu_pct)
        self.memory_percent.append(memory.percent)
        self.memory_gb.append(memory.used / (1024**3))
        self.gpu_memory_allocated.append(gpu_allocated)
        self.gpu_memory_reserved.append(gpu_reserved)
        self.gpu_utilization.append(gpu_util)

    def get_current_stats(self):
        """Get current resource statistics."""
        if not self.timestamps:
            return None

        return {
            "timestamp": self.timestamps[-1],
            "cpu_percent": self.cpu_percent[-1] if self.cpu_percent else 0,
            "memory_percent": self.memory_percent[-1] if self.memory_percent else 0,
            "memory_gb": self.memory_gb[-1] if self.memory_gb else 0,
            "gpu_memory_allocated_gb": (
                self.gpu_memory_allocated[-1] if self.gpu_memory_allocated else 0
            ),
            "gpu_memory_reserved_gb": (
                self.gpu_memory_reserved[-1] if self.gpu_memory_reserved else 0
            ),
            "gpu_utilization": self.gpu_utilization[-1] if self.gpu_utilization else 0,
        }

    def get_summary_stats(self):
        """Get summary statistics for the monitoring period."""
        if not self.timestamps:
            return None

        return {
            "duration_seconds": (
                self.timestamps[-1] - self.timestamps[0]
            ).total_seconds(),
            "samples_collected": len(self.timestamps),
            "cpu_percent": {
                "mean": np.mean(self.cpu_percent),
                "max": np.max(self.cpu_percent),
                "min": np.min(self.cpu_percent),
                "std": np.std(self.cpu_percent),
            },
            "memory_gb": {
                "mean": np.mean(self.memory_gb),
                "max": np.max(self.memory_gb),
                "min": np.min(self.memory_gb),
                "std": np.std(self.memory_gb),
            },
            "gpu_memory_allocated_gb": {
                "mean": np.mean(self.gpu_memory_allocated),
                "max": np.max(self.gpu_memory_allocated),
                "min": np.min(self.gpu_memory_allocated),
                "std": np.std(self.gpu_memory_allocated),
            },
            "gpu_utilization": {
                "mean": np.mean(self.gpu_utilization),
                "max": np.max(self.gpu_utilization),
                "min": np.min(self.gpu_utilization),
                "std": np.std(self.gpu_utilization),
            },
        }

    def save_data(self, output_path):
        """Save monitoring data to JSON file."""

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        data = {
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "cpu_percent": list(self.cpu_percent),
            "memory_percent": list(self.memory_percent),
            "memory_gb": list(self.memory_gb),
            "gpu_memory_allocated_gb": list(self.gpu_memory_allocated),
            "gpu_memory_reserved_gb": list(self.gpu_memory_reserved),
            "gpu_utilization": list(self.gpu_utilization),
            "summary": self.get_summary_stats(),
        }

        # Convert all numpy types to native Python types
        data = convert_numpy_types(data)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Monitoring data saved to {output_path}")
        return output_path

    def plot_usage(self, output_path=None, show_plot=False):
        """Generate plots of resource usage over time."""
        if not self.timestamps:
            logger.warning("No monitoring data available for plotting")
            return None

        # Convert timestamps to datetime objects for plotting
        timestamps = list(self.timestamps)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Resource Usage Over Time", fontsize=16)

        # CPU Usage
        axes[0, 0].plot(timestamps, self.cpu_percent, "b-", linewidth=1)
        axes[0, 0].set_title("CPU Usage (%)")
        axes[0, 0].set_ylabel("CPU %")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Memory Usage
        axes[0, 1].plot(timestamps, self.memory_gb, "r-", linewidth=1)
        axes[0, 1].set_title("System Memory Usage")
        axes[0, 1].set_ylabel("Memory (GB)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # GPU Memory Usage
        if self.gpu_available and self.gpu_memory_allocated:
            axes[1, 0].plot(
                timestamps,
                self.gpu_memory_allocated,
                "g-",
                linewidth=1,
                label="Allocated",
            )
            axes[1, 0].plot(
                timestamps,
                self.gpu_memory_reserved,
                "orange",
                linewidth=1,
                label="Reserved",
            )
            axes[1, 0].set_title("GPU Memory Usage")
            axes[1, 0].set_ylabel("GPU Memory (GB)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis="x", rotation=45)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "GPU not available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("GPU Memory Usage")

        # GPU Utilization
        if self.nvml_available and self.gpu_utilization:
            axes[1, 1].plot(timestamps, self.gpu_utilization, "purple", linewidth=1)
            axes[1, 1].set_title("GPU Utilization")
            axes[1, 1].set_ylabel("GPU Utilization (%)")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis="x", rotation=45)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "GPU utilization not available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("GPU Utilization")

        # Format x-axis
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))

        plt.tight_layout()

        # Save plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Resource usage plot saved to {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return output_path

    def print_summary(self):
        """Print a summary of resource usage."""
        summary = self.get_summary_stats()
        if not summary:
            logger.warning("No monitoring data available")
            return

        print("\n" + "=" * 60)
        print("RESOURCE USAGE SUMMARY")
        print("=" * 60)
        print(f"Monitoring duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Samples collected: {summary['samples_collected']}")
        print()

        print("CPU Usage (%):")
        print(f"  Mean: {summary['cpu_percent']['mean']:.1f}")
        print(f"  Max:  {summary['cpu_percent']['max']:.1f}")
        print(f"  Min:  {summary['cpu_percent']['min']:.1f}")
        print(f"  Std:  {summary['cpu_percent']['std']:.1f}")
        print()

        print("System Memory (GB):")
        print(f"  Mean: {summary['memory_gb']['mean']:.2f}")
        print(f"  Max:  {summary['memory_gb']['max']:.2f}")
        print(f"  Min:  {summary['memory_gb']['min']:.2f}")
        print(f"  Std:  {summary['memory_gb']['std']:.2f}")
        print()

        if self.gpu_available:
            print("GPU Memory Allocated (GB):")
            print(f"  Mean: {summary['gpu_memory_allocated_gb']['mean']:.2f}")
            print(f"  Max:  {summary['gpu_memory_allocated_gb']['max']:.2f}")
            print(f"  Min:  {summary['gpu_memory_allocated_gb']['min']:.2f}")
            print(f"  Std:  {summary['gpu_memory_allocated_gb']['std']:.2f}")
            print()

        if self.nvml_available:
            print("GPU Utilization (%):")
            print(f"  Mean: {summary['gpu_utilization']['mean']:.1f}")
            print(f"  Max:  {summary['gpu_utilization']['max']:.1f}")
            print(f"  Min:  {summary['gpu_utilization']['min']:.1f}")
            print(f"  Std:  {summary['gpu_utilization']['std']:.1f}")
            print()

        print("=" * 60)


def monitor_pipeline_execution(func, *args, output_dir=None, **kwargs):
    """
    Decorator to monitor resource usage during pipeline execution.

    Args:
        func: Function to monitor
        output_dir: Directory to save monitoring results
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Tuple of (function_result, monitor_instance)
    """
    monitor = ResourceMonitor()

    try:
        # Start monitoring
        monitor.start_monitoring()

        # Execute the function
        result = func(*args, **kwargs)

        return result, monitor

    except Exception as e:
        logger.error(f"Error during monitored execution: {e}")
        raise
    finally:
        # Stop monitoring
        monitor.stop_monitoring()

        # Save results if output directory is provided
        if output_dir and os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

            # Save data
            data_path = os.path.join(output_dir, "resource_usage_data.json")
            monitor.save_data(data_path)

            # Generate and save plots
            plot_path = os.path.join(output_dir, "resource_usage_plot.png")
            monitor.plot_usage(plot_path)

            # Print summary
            monitor.print_summary()

            logger.info(f"Resource monitoring results saved to {output_dir}")


# Convenience function for monitoring the flow function
def monitor_flow_execution(
    subject="subject4",
    session="Session0",
    cam="Cam3",
    video="squats1",
    video_name="squats1.avi",
    case_num="case_scaling_fixed",
    rerun=True,
):
    """
    Monitor resource usage during flow execution.
    """
    from flow import flow

    # Create output directory for monitoring results
    output_dir = os.path.join(
        "output", case_num, subject, session, cam, video, "monitoring"
    )

    # Execute flow with monitoring
    result, monitor = monitor_pipeline_execution(
        flow,
        subject,
        session,
        cam,
        video,
        video_name,
        case_num,
        rerun,
        output_dir=output_dir,
    )

    return result, monitor


if __name__ == "__main__":
    # Test the monitoring system
    monitor = ResourceMonitor(sampling_interval=0.5)

    print("Starting resource monitoring test...")
    monitor.start_monitoring()

    # Simulate some work
    time.sleep(10)

    monitor.stop_monitoring()

    # Print summary and generate plots
    monitor.print_summary()
    monitor.plot_usage("test_resource_usage.png", show_plot=True)
