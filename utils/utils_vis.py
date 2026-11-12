#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 23:59:46 2024

@author: opencap
"""

# Plot keypoints

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import os
import plotly.graph_objects as go


def plot_2d_keypoints_interactive(
    keypoints, image_w=720, image_h=1280, save_path=None, show=True, save_html=True
):
    # Determine the number of time frames
    T = next(iter(keypoints.values())).shape[0]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)

    # Set fixed axis limits and reverse y-axis
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)  # Flip y-axis

    # Adjust aspect ratio
    ax.set_aspect(abs((image_w / ax.get_xlim()[1]) / (image_h / ax.get_ylim()[0])))

    # Store scatter plot objects
    scatter_plots = {}

    # Initial plot
    for key_label, points in keypoints.items():
        scatter_plots[key_label] = ax.scatter(
            points[0, :, 0], points[0, :, 1], label=key_label, s=5
        )

    ax.legend()

    # Update function
    def update(t=0):
        for key_label, scatter_plot in scatter_plots.items():
            scatter_plot.set_offsets(
                np.c_[keypoints[key_label][t, :, 0], keypoints[key_label][t, :, 1]]
            )
        fig.canvas.draw_idle()

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, "Time", 0, T - 1, valinit=0, valstep=1)

    slider.on_changed(update)

    if show:
        plt.show()
    if save_path is not None:
        if save_html:
            fig_path = os.path.join(save_path, "2d_keypoints_plot.html")
        else:
            fig_path = os.path.join(save_path, "2d_keypoints_plot.png")
        fig.savefig(fig_path)
        # logger.info(f"2D keypoints plot saved at {fig_path}")

    return slider, fig_path


def plot_3d_keypoints_interactive(key3d_dict, save_path=None, show=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.25)

    # Calculate the common axis range
    all_data = np.concatenate(list(key3d_dict.values()))
    min_value = np.min(all_data)
    max_value = np.max(all_data)

    # Set the same range for each axis
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_zlim(min_value, max_value)

    # Create a color map for different keys
    colors = plt.cm.jet(np.linspace(0, 1, len(key3d_dict)))

    # Initial plot with legend
    lines = []
    for (key, key3d), color in zip(key3d_dict.items(), colors):
        line = ax.scatter(
            key3d[0, :, 0], key3d[0, :, 1], key3d[0, :, 2], color=color, label=key
        )
        lines.append(line)
    ax.legend()

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, "Time", 0, all_data.shape[0] - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        t = int(slider.val)
        ax.clear()

        for key, key3d in key3d_dict.items():
            ax.scatter(key3d[t, :, 0], key3d[t, :, 1], key3d[t, :, 2], label=key)

        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)
        ax.set_zlim(min_value, max_value)

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.legend()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    if show:
        plt.show()
    if save_path is not None:
        fig_path = os.path.join(save_path, "3d_keypoints_plot.png")
        fig.savefig(fig_path)
        # logger.info(f"3D keypoints plot saved at {fig_path}")

    return slider


def plot_objective_function(output_opt, save_path=None, show=False):
    # Extract the objective values
    objective_values = (
        output_opt["objective_values"].cpu().numpy()
    )  # Convert to NumPy array

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the objective values
    ax.plot(
        range(len(objective_values)),
        objective_values,
        label="Objective Function",
        linestyle="-",
        marker="",
    )

    # Customize the plot
    ax.set_title("Objective Function Plot")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.legend()
    ax.grid(True)

    # Save the plot if a save_path is provided
    if save_path is not None:
        fig_path = os.path.join(save_path, "objective_function_plot.png")
        plt.savefig(fig_path, dpi=300)
        # logger.info(f"Objective function plot saved at {fig_path}")

    # Show the plot if required
    if show:
        plt.show()

    # Close the figure to free memory if not showing
    if not show:
        plt.close(fig)


def plot_2d_keypoints_interactive_plotly(
    keypoints,
    image_w=720,
    image_h=1280,
    save_path=None,
    range_mono=None,
    fig_show=False,
):
    # Predefined color palette
    color_palette = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "brown",
        "pink",
        "gray",
        "olive",
        "lime",
    ]

    # Assign colors to key_labels
    key_label_colors = {
        key_label: color_palette[i % len(color_palette)]
        for i, key_label in enumerate(keypoints.keys())
    }

    # Slice keypoints if range_mono is specified
    if range_mono is not None:
        start, end = range_mono[0], range_mono[-1] + 1  # Include the last frame
        keypoints = {
            key_label: points[start:end]  # Slice along the first dimension (time)
            for key_label, points in keypoints.items()
        }

    # Determine the maximum number of frames across all datasets
    T = max(points.shape[0] for points in keypoints.values())

    # Create the figure and frames
    frames = []

    for t in range(T):
        frame_data = []
        for key_label, points in keypoints.items():
            if t < points.shape[0]:  # Ensure frame exists after slicing
                frame_data.append(
                    go.Scatter(
                        x=points[t, :, 0],  # X-coordinate
                        y=points[t, :, 1],  # Y-coordinate
                        mode="markers",
                        marker=dict(size=8, color=key_label_colors[key_label]),
                        name=key_label,
                        text=[
                            f"Index: {i}<br>Coords: ({x:.2f}, {y:.2f})<br>Source: {key_label}"
                            for i, (x, y) in enumerate(points[t, :, :2])
                        ],
                        hoverinfo="text",
                    )
                )
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Add initial data for frame 0
    initial_data = []
    for key_label, points in keypoints.items():
        if points.shape[0] > 0:  # Ensure at least one frame exists
            initial_data.append(
                go.Scatter(
                    x=points[0, :, 0],
                    y=points[0, :, 1],
                    mode="markers",
                    marker=dict(size=8, color=key_label_colors[key_label]),
                    name=key_label,
                    text=[
                        f"Index: {i}<br>Coords: ({x:.2f}, {y:.2f})<br>Source: {key_label}"
                        for i, (x, y) in enumerate(points[0, :, :2])
                    ],
                    hoverinfo="text",
                )
            )

    # Add data and frames to the figure
    fig = go.Figure(data=initial_data, frames=frames)

    # Add slider and playback controls
    fig.update_layout(
        xaxis=dict(range=[0, image_w], title="X"),
        yaxis=dict(range=[image_h, 0], title="Y"),  # Flip y-axis
        title="2D Keypoints with Frame Slider",
        width=800,
        height=600,
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(t),
                        "method": "animate",
                    }
                    for t in range(T)
                ],
                "currentvalue": {"prefix": "Frame: ", "font": {"size": 16}},
                "pad": {"t": 50},
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                ],
            }
        ],
    )

    # Save the interactive plot as HTML
    if save_path:
        html_path = f"{save_path}/2d_keypoints_plot.html"
        fig.write_html(html_path)
        # print(f"Interactive plot saved to {html_path}")

    if fig_show:
        fig.show()

    return html_path if save_path else None


def plot_3d_keypoints_interactive_plotly(
    keypoints3d,
    image_w=720,
    image_h=1280,
    save_path=None,
    range_mono=None,
    fig_show=False,
):
    """
    Create an interactive Plotly plot for 3D keypoints with a frame slider.

    Parameters:
    - keypoints3d: dict mapping key_labels to 3D arrays of shape (T, N, 3)
    - image_w: int, width of the image (used for axis limits)
    - image_h: int, height of the image (used for axis limits)
    - save_path: str or None, directory to save the HTML file (optional)
    - range_mono: list or tuple or None, range of frames to include (start, end), inclusive
    """
    # TODO work on this function because it does not plot what we want. Need to understand the keypoints3d numbers better
    # Slice keypoints if range_mono is specified
    if range_mono is not None:
        start, end = range_mono[0], range_mono[-1] + 1  # Include the last frame
        keypoints3d = {
            key_label: points[start:end]  # Slice along the first dimension (time)
            for key_label, points in keypoints3d.items()
        }

    # Determine the maximum number of frames across all datasets
    T = max(points.shape[0] for points in keypoints3d.values())

    # Create the figure and frames
    fig = go.Figure()
    frames = []

    # Initialize axis limits and view settings
    x_range = [0, image_w]
    y_range = [
        0,
        max(max(points[:, :, 1].max() for points in keypoints3d.values()), 1),
    ]  # Adjust y to be up
    z_range = [0, image_h]  # Adjust z to be horizontal
    camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Adjust to set a fixed point of view

    for t in range(T):
        frame_data = []
        for key_label, points in keypoints3d.items():
            if t < points.shape[0]:  # Ensure frame exists after slicing
                frame_data.append(
                    go.Scatter3d(
                        x=points[t, :, 0],  # X-coordinate
                        y=points[t, :, 2],  # Adjust Y to Z-coordinate (up direction)
                        z=points[t, :, 1],  # Adjust Z to Y-coordinate (horizontal)
                        mode="markers",
                        marker=dict(size=5),
                        name=key_label,
                        text=[
                            f"Index: {i}<br>Coords: ({x:.2f}, {y:.2f}, {z:.2f})<br>Source: {key_label}"
                            for i, (x, y, z) in enumerate(points[t, :, :])
                        ],
                        hoverinfo="text",
                    )
                )
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Add initial data for frame 0
    initial_data = []
    for key_label, points in keypoints3d.items():
        if points.shape[0] > 0:  # Ensure at least one frame exists
            initial_data.append(
                go.Scatter3d(
                    x=points[0, :, 0],
                    y=points[0, :, 2],  # Adjust Y to Z-coordinate (up direction)
                    z=points[0, :, 1],  # Adjust Z to Y-coordinate (horizontal)
                    mode="markers",
                    marker=dict(size=5),
                    name=key_label,
                    text=[
                        f"Index: {i}<br>Coords: ({x:.2f}, {y:.2f}, {z:.2f})<br>Source: {key_label}"
                        for i, (x, y, z) in enumerate(points[0, :, :])
                    ],
                    hoverinfo="text",
                )
            )

    # Add data and frames to the figure
    fig = go.Figure(data=initial_data, frames=frames)

    # Add slider and playback controls
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range, title="X"),
            yaxis=dict(range=y_range, title="Z"),  # Update axis title
            zaxis=dict(range=z_range, title="Y"),  # Update axis title
            camera=camera,
        ),
        title="3D Keypoints with Frame Slider",
        width=800,
        height=600,
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(t),
                        "method": "animate",
                    }
                    for t in range(T)
                ],
                "currentvalue": {"prefix": "Frame: ", "font": {"size": 16}},
                "pad": {"t": 50},
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            }
        ],
    )

    # Save the interactive plot as HTML
    if save_path:
        html_path = f"{save_path}/3d_keypoints_plot.html"
        fig.write_html(html_path)
        # print(f"Interactive plot saved to {html_path}")

    if fig_show:
        fig.show()

    return html_path if save_path else None
