"""Manages the movement and use of data files."""

import os
import warnings
from scipy.spatial.transform import Rotation as R

import numpy as np
from numpy.lib.recfunctions import append_fields
from loguru import logger


class TRCFile(object):
    """A plain-text file format for storing motion capture marker trajectories.
    TRC stands for Track Row Column.

    The metadata for the file is stored in attributes of this object.

    See
    http://simtk-confluence.stanford.edu:8080/display/OpenSim/Marker+(.trc)+Files
    for more information.

    """

    def __init__(self, fpath=None, **kwargs):
        # path=None,
        # data_rate=None,
        # camera_rate=None,
        # num_frames=None,
        # num_markers=None,
        # units=None,
        # orig_data_rate=None,
        # orig_data_start_frame=None,
        # orig_num_frames=None,
        # marker_names=None,
        # time=None,
        # ):
        """
        Parameters
        ----------
        fpath : str
            Valid file path to a TRC (.trc) file.

        """
        self.marker_names = []
        if fpath != None:
            self.read_from_file(fpath)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def copy(self):
        """Create a copy of the TRCFile object."""
        return TRCFile(
            marker_names=self.marker_names.copy(),
            data_rate=self.data_rate,
            camera_rate=self.camera_rate,
            num_frames=self.num_frames,
            num_markers=self.num_markers,
            units=self.units,
            orig_data_rate=self.orig_data_rate,
            orig_data_start_frame=self.orig_data_start_frame,
            orig_num_frames=self.orig_num_frames,
            data=self.data.copy(),
            time=self.time.copy(),
        )

    def read_from_file(self, fpath):
        # Read the header lines / metadata.
        # ---------------------------------
        # Split by any whitespace.
        # TODO may cause issues with paths that have spaces in them.
        f = open(fpath)
        # These are lists of each entry on the first few lines.
        first_line = f.readline().split()
        second_line = f.readline().split()
        third_line = f.readline().split()
        fourth_line = f.readline().split()
        f.close()

        # Check if first line is header (contains "DataRate") or old format (path)
        if len(first_line) > 0 and first_line[0] == "DataRate":
            # New format: Line 1 = header names, Line 2 = header values
            # First line.
            if len(first_line) > 3:
                self.path = first_line[3] if len(first_line) > 3 else ""
            else:
                self.path = ""
            
            # Second line contains the actual values
            try:
                self.data_rate = float(second_line[0])
                self.camera_rate = float(second_line[1])
                self.num_frames = int(second_line[2])
                self.num_markers = int(second_line[3])
                self.units = second_line[4] if len(second_line) > 4 else "m"
                self.orig_data_rate = float(second_line[5]) if len(second_line) > 5 else self.data_rate
                self.orig_data_start_frame = int(second_line[6]) if len(second_line) > 6 else 1
                self.orig_num_frames = int(second_line[7]) if len(second_line) > 7 else self.num_frames
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing TRC header values from line 2: {second_line}. Error: {e}")
        else:
            # Old format: Line 1 = optional path, Line 2 = skip, Line 3 = values
            # First line.
            if len(first_line) > 3:
                self.path = first_line[3]
            else:
                self.path = ""

            # Third line.
            self.data_rate = float(third_line[0])
            self.camera_rate = float(third_line[1])
            self.num_frames = int(third_line[2])
            self.num_markers = int(third_line[3])
            self.units = third_line[4]
            self.orig_data_rate = float(third_line[5])
            self.orig_data_start_frame = int(third_line[6])
            self.orig_num_frames = int(third_line[7])

        # Marker names.
        # The first and second column names are 'Frame#' and 'Time'.
        # In new format, marker names are on line 3; in old format, on line 4
        if len(first_line) > 0 and first_line[0] == "DataRate":
            # New format: marker names on third_line
            self.marker_names = third_line[2:]
        else:
            # Old format: marker names on fourth_line
            self.marker_names = fourth_line[2:]

        len_marker_names = len(self.marker_names)
        if len_marker_names != self.num_markers:
            warnings.warn(
                "Header entry NumMarkers, %i, does not "
                "match actual number of markers, %i. Changing "
                "NumMarkers to match actual number."
                % (self.num_markers, len_marker_names)
            )
            self.num_markers = len_marker_names

        # Load the actual data.
        # ---------------------
        col_names = ["frame_num", "time"]
        # This naming convention comes from OpenSim's Inverse Kinematics tool,
        # when it writes model marker locations.
        for mark in self.marker_names:
            col_names += [mark + "_tx", mark + "_ty", mark + "_tz"]
        dtype = {
            "names": col_names,
            "formats": ["int"] + ["float64"] * (3 * self.num_markers + 1),
        }
        usecols = [i for i in range(3 * self.num_markers + 1 + 1)]
        self.data = np.loadtxt(
            fpath, delimiter="\t", skiprows=5, dtype=dtype, usecols=usecols
        )
        self.time = self.data["time"]

        # Check the number of rows.
        n_rows = self.time.shape[0]
        if n_rows != self.num_frames:
            warnings.warn(
                "%s: Header entry NumFrames, %i, does not "
                "match actual number of frames, %i, Changing "
                "NumFrames to match actual number." % (fpath, self.num_frames, n_rows)
            )
            self.num_frames = n_rows

    def __getitem__(self, key):
        """See `marker()`."""
        return self.marker(key)

    def units(self):
        return self.units

    def time(self):
        this_dat = np.empty((self.num_frames, 1))
        this_dat[:, 0] = self.time
        return this_dat

    def marker(self, name):
        """The trajectory of marker `name`, given as a `self.num_frames` x 3
        array. The order of the columns is x, y, z.

        """
        this_dat = np.empty((self.num_frames, 3))
        this_dat[:, 0] = self.data[name + "_tx"]
        this_dat[:, 1] = self.data[name + "_ty"]
        this_dat[:, 2] = self.data[name + "_tz"]
        return this_dat

    def add_marker(self, name, x, y, z):
        """Add a marker, with name `name` to the TRCFile.

        Parameters
        ----------
        name : str
            Name of the marker; e.g., 'R.Hip'.
        x, y, z: array_like
            Coordinates of the marker trajectory. All 3 must have the same
            length.

        """
        if (
            len(x) != self.num_frames
            or len(y) != self.num_frames
            or len(z) != self.num_frames
        ):
            raise Exception(
                "Length of data (%i, %i, %i) is not " "NumFrames (%i).",
                len(x),
                len(y),
                len(z),
                self.num_frames,
            )
        self.marker_names += [name]
        self.num_markers += 1
        if not hasattr(self, "data"):
            self.data = np.array(x, dtype=[("%s_tx" % name, "float64")])
            self.data = append_fields(
                self.data, ["%s_t%s" % (name, s) for s in "yz"], [y, z], usemask=False
            )
        else:
            self.data = append_fields(
                self.data,
                ["%s_t%s" % (name, s) for s in "xyz"],
                [x, y, z],
                usemask=False,
            )

    def marker_at(self, name, time):
        x = np.interp(time, self.time, self.data[name + "_tx"])
        y = np.interp(time, self.time, self.data[name + "_ty"])
        z = np.interp(time, self.time, self.data[name + "_tz"])
        return [x, y, z]

    def marker_exists(self, name):
        """
        Returns
        -------
        exists : bool
            Is the marker in the TRCFile?

        """
        return name in self.marker_names

    def write(self, fpath):
        """Write this TRCFile object to a TRC file.

        Parameters
        ----------
        fpath : str
            Valid file path to which this TRCFile is saved.

        """
        f = open(fpath, "w")

        # Line 1.
        f.write("PathFileType  4\t(X/Y/Z) %s\n" % os.path.split(fpath)[0])

        # Line 2.
        f.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
            "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )

        # Line 3.
        f.write(
            "%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n"
            % (
                self.data_rate,
                self.camera_rate,
                self.num_frames,
                self.num_markers,
                self.units,
                self.orig_data_rate,
                self.orig_data_start_frame,
                self.orig_num_frames,
            )
        )

        # Line 4.
        f.write("Frame#\tTime\t")
        for imark in range(self.num_markers):
            f.write("%s\t\t\t" % self.marker_names[imark])
        f.write("\n")

        # Line 5.
        f.write("\t\t")
        for imark in np.arange(self.num_markers) + 1:
            f.write("X%i\tY%s\tZ%s\t" % (imark, imark, imark))
        f.write("\n")

        # Line 6.
        f.write("\n")

        # Data.
        for iframe in range(self.num_frames):
            f.write("%i" % (iframe + 1))
            f.write("\t%.7f" % self.time[iframe])
            for mark in self.marker_names:
                idxs = [mark + "_tx", mark + "_ty", mark + "_tz"]
                f.write(
                    "\t%.7f\t%.7f\t%.7f"
                    % tuple(self.data[coln][iframe] for coln in idxs)
                )
            f.write("\n")

        f.close()

    def add_noise(self, noise_width):
        """add random noise to each component of the marker trajectory
        The noise mean will be zero, with the noise_width being the
        standard deviation.

        noise_width : int
        """
        for imarker in range(self.num_markers):
            components = ["_tx", "_ty", "_tz"]
            for iComponent in range(3):
                # generate noise
                noise = np.random.normal(0, noise_width, self.num_frames)
                # add noise to each component of marker data.
                self.data[self.marker_names[imarker] + components[iComponent]] += noise

    def rotate(self, axis, value):
        """rotate the data.

        axis : rotation axis
        value : angle in degree
        """
        for imarker in range(self.num_markers):
            temp = np.zeros((self.num_frames, 3))
            temp[:, 0] = self.data[self.marker_names[imarker] + "_tx"]
            temp[:, 1] = self.data[self.marker_names[imarker] + "_ty"]
            temp[:, 2] = self.data[self.marker_names[imarker] + "_tz"]

            r = R.from_euler(axis, value, degrees=True)
            temp_rot = r.apply(temp)

            self.data[self.marker_names[imarker] + "_tx"] = temp_rot[:, 0]
            self.data[self.marker_names[imarker] + "_ty"] = temp_rot[:, 1]
            self.data[self.marker_names[imarker] + "_tz"] = temp_rot[:, 2]

    def offset(self, axis, value, single_marker=None):
        """offset the data.

        axis : rotation axis
        value : offset in m
        """
        if single_marker is not None:
            if single_marker not in self.marker_names:
                raise ValueError("Marker not found")
            else:
                marker_idx = self.marker_names.index(single_marker)
                if axis.lower() == "x":
                    self.data[self.marker_names[marker_idx] + "_tx"] += value
                elif axis.lower() == "y":
                    self.data[self.marker_names[marker_idx] + "_ty"] += value
                elif axis.lower() == "z":
                    self.data[self.marker_names[marker_idx] + "_tz"] += value
                else:
                    raise ValueError("Axis not recognized")
        else:
            for imarker in range(self.num_markers):
                if axis.lower() == "x":
                    self.data[self.marker_names[imarker] + "_tx"] += value
                elif axis.lower() == "y":
                    self.data[self.marker_names[imarker] + "_ty"] += value
                elif axis.lower() == "z":
                    self.data[self.marker_names[imarker] + "_tz"] += value
                else:
                    raise ValueError("Axis not recognized")

    def resample_trc(self, target_frequency=100):
        """
        Resample the TRC data to a target frequency.

        Parameters:
            trc_mono: TRCFile
                The TRC object containing the data and time information.
            target_frequency: int
                The desired frequency (e.g., 100 for 100 Hz).

        Returns:
            dict
                A dictionary containing the resampled time and data.
        """
        from scipy.interpolate import interp1d

        original_data = self.data
        original_time = original_data["time"]

        # Define new time vector based on the target frequency
        start_time = original_time[0]
        end_time = original_time[-1]
        new_time = np.arange(start_time, end_time, 1 / target_frequency)

        # Extract numerical data for interpolation (excluding frame_num and time)
        numeric_fields = [
            field
            for field in original_data.dtype.names
            if field not in ["frame_num", "time"]
        ]
        numeric_data = np.vstack([original_data[field] for field in numeric_fields]).T

        # Interpolate the entire array at once
        f_interp = interp1d(
            original_time, numeric_data, axis=0, kind="linear", fill_value="extrapolate"
        )
        interpolated_numeric_data = f_interp(new_time)

        # Combine frame numbers, time, and interpolated numeric data
        frame_numbers = np.arange(1, len(new_time) + 1)  # Generate new frame numbers
        combined_data = np.column_stack(
            (frame_numbers, new_time, interpolated_numeric_data)
        )

        # Convert each row to a tuple
        dtype = [
            (name, original_data.dtype[name]) for name in original_data.dtype.names
        ]
        interpolated_data_tuples = np.array(
            [tuple(row) for row in combined_data], dtype=dtype
        )

        self.data = interpolated_data_tuples
        self.camera_rate = target_frequency
        self.data_rate = int(target_frequency)
        self.time = new_time
        self.num_frames = len(new_time)
        # TODO check if this is needed
        # self.orig_num_frames = len(new_time)
        # self.orig_data_rate = target_frequency
        # self.orig_data_start_frame = 1

    def convert_to_metric_trc(self, current_metric, target_metric):
        """
        Convert all marker position columns in a TRC NumPy structured array
        from the original metric to the target metric.

        Parameters:
            current_metric : str
                The current metric of the data ('mm', 'cm', 'm').
            target_metric : str
                The target metric to convert the data to ('mm', 'cm', 'm').

        Returns:
            None
        """
        if not hasattr(self, "data") or not isinstance(self.data, np.ndarray):
            raise ValueError(
                "TRC object must have a 'data' attribute as a NumPy structured array."
            )

        data = self.data
        conversion_factors = {
            ("mm", "cm"): 0.1,
            ("mm", "m"): 0.001,
            ("cm", "mm"): 10,
            ("cm", "m"): 0.01,
            ("m", "mm"): 1000,
            ("m", "cm"): 100,
        }

        if current_metric == target_metric:
            logger.info(
                "Current metric is already the target metric. No conversion performed."
            )
            return

        factor = conversion_factors.get((current_metric, target_metric))
        if factor is None:
            raise ValueError(
                f"Unsupported conversion from {current_metric} to {target_metric}"
            )

        # Identify marker columns (exclude 'time' and 'frame_num')
        marker_columns = [
            col for col in data.dtype.names if col not in ["frame_num", "time"]
        ]

        # Apply conversion factor to marker columns
        for col in marker_columns:
            data[col] *= factor

        logger.info(f"Converted {current_metric} to {target_metric}.")
        self.units = target_metric

    def get_frequency(self):
        """
        Get the frequency of the TRC data.

        Returns:
            int
                The frequency of the data in Hz.
        """
        return int(self.data_rate)

    def get_metric_trc(self):
        """
        Determine the metric of the TRC data based on marker positions.

        Returns:
            str
                The detected metric of the data ('mm', 'cm', 'm').
        """
        # get the right knee position at frame 0
        right_knee = self.marker("r_knee")
        right_knee = np.max(right_knee[0])
        if right_knee > 100:
            metric = "mm"
        elif right_knee > 10:
            metric = "cm"
        else:
            metric = "m"

        return metric

    # def shift_data_by_lag(self, lag, target_length):
    #     """
    #     Shift the TRC data based on a given lag and update self.data.
    #
    #     Parameters:
    #         lag : int
    #             The lag in frames. Positive values indicate a forward shift,
    #             and negative values indicate a backward shift.
    #         target_length : int
    #             The length of the resulting data after the shift (e.g., mocap data length).
    #
    #     Returns:
    #         None
    #     """
    #     num_frames = self.data.shape[0]  # Total number of frames in the current data
    #
    #     if lag > 0:
    #         # start_idx = lag
    #         end_idx = min(num_frames + lag, target_length)
    #         self.data = self.data[:end_idx - lag]
    #     elif lag < 0:
    #         # start_idx = 0
    #         end_idx = min(num_frames, target_length + lag)
    #         self.data = self.data[-lag:end_idx - lag]
    #     else:
    #         # start_idx = 0
    #         end_idx = min(num_frames, target_length)
    #         self.data = self.data[:end_idx]
    #
    #     # Ensure self.data retains the same dtype
    #     self.data = np.array(self.data, dtype=self.data.dtype)

    def get_start_end_times(self):
        start = self.time[0]
        end = self.time[-1]

        return start, end

    def add_time_offset(self, offset):
        self.time += offset
        self.data["time"] += offset

    def trim_to_match(self, start_time, end_time):
        """
        Trim the TRC data to match the start and end times.

        Parameters:
            start_time : float
                The start time in seconds.
            end_time : float
                The end time in seconds.

        Returns:
            None
        """
        # Find the indices of the start and end times
        start_idx = np.argmin(np.abs(self.time - start_time))
        if end_time > self.time[-1]:
            end_idx = len(self.time) - 1
        else:
            end_idx = np.argmin(np.abs(self.time - end_time))

        # Trim the data
        self.data = self.data[start_idx : end_idx + 1]
        self.time = self.time[start_idx : end_idx + 1]
        self.num_frames = len(self.data)
        self.num_markers = len(self.marker_names)
        self.orig_num_frames = len(self.data)
        self.orig_data_start_frame = 1

    def get_marker_names(self):
        return self.marker_names

    def tuple_to_numpy(self):
        # Convert from ndarray (num_frames,) where each element is a tuple of num_markers x 3 to np array (num_frames, num_markers*3). Skip 'frame_num' and 'time'
        data = np.zeros((self.num_frames, self.num_markers * 3))
        for i, row in enumerate(self.data):
            concatenated_row = np.concatenate(
                [
                    np.array(row[col])
                    for col in self.data.dtype.names
                    if col not in ["frame_num", "time"]
                ]
            )
            data[i] = concatenated_row
        return data

    # def adjust_and_slice_by_lag(self, lag, target_length):
    #     """
    #     Adjust and slice TRC data based on a given lag to match a target length.
    #
    #     Parameters:
    #         lag : int
    #             The lag in frames. Positive values indicate a forward shift,
    #             and negative values indicate a backward shift.
    #         target_length : int
    #             The length of the resulting data after the adjustment.
    #
    #     Returns:
    #         None
    #     """
    #     num_frames = self.num_frames  # Total number of frames in the current TRC data
    #
    #     if lag > 0:
    #         # Adjust start and end indices
    #         start_idx = lag
    #         end_idx = min(num_frames + lag, target_length)
    #         # Slice the data and reset
    #         self.data = self.data[start_idx:end_idx]
    #         self.time = self.time[start_idx:end_idx]
    #     elif lag < 0:
    #         # Adjust start and end indices
    #         start_idx = 0
    #         end_idx = min(num_frames, target_length + lag)
    #         # Slice the data and reset
    #         self.data = self.data[-lag:end_idx - lag]
    #         self.time = self.time[-lag:end_idx - lag]
    #     else:
    #         # No lag; simply trim to the target length
    #         start_idx = 0
    #         end_idx = min(num_frames, target_length)
    #         self.data = self.data[start_idx:end_idx]
    #         self.time = self.time[start_idx:end_idx]
    #
    #     # Update the frame count to reflect the new length
    #     self.num_frames = len(self.data)

    # def adjust_and_slice_by_lag(self, lag, trc_mocap):
    #     """
    #     Adjust TRC data by finding the time of trc_mocap at the lag value and adding this time offset to all time values of trc_mono.
    #
    #     Parameters:
    #         lag : int
    #             The lag in frames. Positive values indicate a forward shift,
    #             and negative values indicate a backward shift.
    #         trc_mocap : TRCFile
    #             The TRCFile object containing the mocap data.
    #
    #     Returns:
    #         None
    #     """
    #     # Find the time offset from trc_mocap at the lag value
    #     if lag > 0:
    #         time_offset = trc_mocap.time[lag]
    #     elif lag < 0:
    #         time_offset = trc_mocap.time[lag]
    #     else:
    #         time_offset = 0
    #
    #     # Add the time offset to all time values of trc_mono
    #     self.time += time_offset
    #     self.data['time'] += time_offset
    #
    #     # Update the frame count to reflect the new length
    #     self.num_frames = len(self.data)


def align_trc_files(trc_mono, trc_mocap, lag):
    """
    Aligns trc_mono to trc_mocap by adjusting the time vector based on lag.

    Parameters:
        trc_mono : TRCFile
            The TRC object for mono data.
        trc_mocap : TRCFile
            The TRC object for mocap data.
        lag : int
            The lag value (in frames) between trc_mono and trc_mocap.

    Returns:
        TRCFile
            A new TRCFile object for the aligned trc_mono.
    """
    # Get the time offset from mocap time at trc_mono time 0 adjusted by lag
    mocap_start_time = trc_mocap.time[0]
    mono_start_time_adjusted = trc_mono.time[0] + lag / trc_mono.data_rate
    time_offset = mono_start_time_adjusted - mocap_start_time

    # Shift the time vector of trc_mono
    trc_mono.add_time_offset(time_offset)

    # round the time values to 3 decimal places
    trc_mono.time = np.round(trc_mono.time, 3)

    # Trim trc_mono to match the time range of trc_mocap
    mocap_start, mocap_end = trc_mocap.get_start_end_times()
    trc_mono.trim_to_match(mocap_start, mocap_end)

    # Return the adjusted trc_mono
    return trc_mono


# Standalone Functions
def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    # data -> nFrames x nMarkers*3 array

    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count + 1] = header

        # Line 1.
    f.write("PathFileType  4\t(X/Y/Z) %s\n" % os.getcwd())

    # Line 2.
    f.write(
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
        "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
    )

    num_frames = data.shape[0]
    num_markers = len(header_mapping.keys())

    # Line 3.
    f.write(
        "%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n"
        % (fc, fc, num_frames, num_markers, units, fc, 1, num_frames)
    )

    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write("X%i\tY%s\tZ%s\t" % (imark, imark, imark))
    f.write("\n")

    # Line 6.
    f.write("\n")

    for frame in range(data.shape[0]):
        f.write(
            "{}\t{:.8f}\t".format(frame + 1, (frame) / fc + t_start)
        )  # opensim frame labeling is 1 indexed

        for key in sorted(header_mapping.keys()):
            f.write(
                "{:.5f}\t{:.5f}\t{:.5f}\t".format(
                    data[frame, 0 + (key - 1) * 3],
                    data[frame, 1 + (key - 1) * 3],
                    data[frame, 2 + (key - 1) * 3],
                )
            )
        f.write("\n")


def write_trc(
    keypoints3D,
    pathOutputFile,
    keypointNames,
    frameRate=60,
    rotationAngles={},
    unit="m",
    t_start=0.0,
    vertical_offset=None,
):
    """
    Write 3D keypoints data to a TRC file.

    Parameters:
    -----------
    keypoints3D : np.ndarray
        A NumPy array containing the 3D coordinates of keypoints. The shape of this array should be (num_frames, num_markers * 3), where each marker has three coordinates (X, Y, Z) for each frame.
    pathOutputFile : str
        The file path where the TRC file will be written.
    keypointNames : list of str
        A list of strings representing the names of the keypoints (markers).
    frameRate : int, optional
        The frame rate of the data. Default is 60.
    rotationAngles : dict, optional
        A dictionary where the keys are axis names ('x', 'y', 'z') and the values are the rotation angles in degrees. This is used to rotate the data to match OpenSim conventions.
    vertical_offset : float, optional
        An optional vertical offset to apply to the Y coordinates of the keypoints.
    Returns:
    --------
    None
    """
    with open(pathOutputFile, "w") as f:
        numpy2TRC(
            f, keypoints3D, keypointNames, fc=frameRate, units=unit, t_start=t_start
        )

    # Rotate data to match OpenSim conventions; this assumes the chessboard
    # is behind the subject and the chessboard axes are parallel to those of
    # OpenSim.
    trc_file = TRCFile(pathOutputFile)
    for axis, angle in rotationAngles.items():
        trc_file.rotate(axis, angle)

    if vertical_offset is not None:
        trc_file.offset("y", -vertical_offset + 0.01)
        # logg the vertical offset
        logger.info(f"Vertical offset: {-vertical_offset + 0.01}")

    trc_file.write(pathOutputFile)

    return None


def transform_to_tuple_array(interpolated_data, frame_numbers, time):
    """
    Transform interpolated data into an ndarray where each element is a tuple.

    Parameters:
        interpolated_data: ndarray
            A 2D NumPy array of shape (num_frames, num_columns).
        frame_numbers: ndarray
            A 1D NumPy array of frame numbers.
        time: ndarray
            A 1D NumPy array of time values corresponding to each frame.

    Returns:
        ndarray
            A 1D NumPy array where each element is a tuple (frame#, time, data...).
    """
    # Combine frame numbers, time, and interpolated data into a single array
    combined_data = np.hstack(
        (
            frame_numbers[:, None],  # Frame numbers as a column
            time[:, None],  # Time as a column
            interpolated_data,  # Marker data
        )
    )

    # Convert each row into a tuple
    tuple_array = np.array([tuple(row) for row in combined_data])

    return tuple_array


def transform_from_tuple_array(void_array):
    """
    Transform a tuple array into separate NumPy arrays for frame numbers, time, and data.

    Parameters:
        tuple_array: ndarray
            A 1D NumPy array where each element is a tuple (frame#, time, data...).

    Returns:
        frame_numbers: ndarray
            A 1D NumPy array of frame numbers.
        time: ndarray
            A 1D NumPy array of time values corresponding to each frame.
        interpolated_data: ndarray
            A 2D NumPy array of shape (num_frames, num_columns).
    """
    # Extract frame numbers and time directly
    frame_numbers = np.array([row[0] for row in void_array])
    time = np.array([row[1] for row in void_array])

    # Extract marker data by iterating over the fields beyond index 1
    data = np.array([tuple(row[i] for i in range(2, len(row))) for row in void_array])
    return data


def read_trc(trc_path):
    """
    Read a TRC file and return the marker data, marker names, and frame rate.

    Parameters:
    -----------
    trc_path : str
        Path to the TRC file

    Returns:
    --------
    marker_data : numpy.ndarray
        Marker positions (n_frames, n_markers, 3)
    marker_names : list
        List of marker names
    frame_rate : float
        Frame rate of the marker data
    """
    trc_file = TRCFile(trc_path)

    # Extract marker names
    marker_names = trc_file.marker_names

    # Extract frame rate
    frame_rate = trc_file.data_rate

    # Extract marker data
    n_frames = trc_file.num_frames
    n_markers = trc_file.num_markers

    # Initialize marker data array
    marker_data = np.zeros((n_frames, n_markers, 3))

    # Fill marker data array
    for i, marker_name in enumerate(marker_names):
        marker_data[:, i, 0] = trc_file.data[marker_name + "_tx"]
        marker_data[:, i, 1] = trc_file.data[marker_name + "_ty"]
        marker_data[:, i, 2] = trc_file.data[marker_name + "_tz"]

    return marker_data, marker_names, frame_rate
