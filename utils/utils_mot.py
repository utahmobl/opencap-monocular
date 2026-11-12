import numpy as np
import pandas as pd
import opensim


# %%  Storage file to numpy array.
def mot_to_numpy(mot_file, excess_header_entries=0):
    """Returns the data from a mot file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    mot_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the mot file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the mot file can be obtained as follows:
        # >>> data = mot2numpy('<filename>')
        # >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(mot_file, "r")

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count("endheader") != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(mot_file, names=names, skip_header=skip_header)

    return data


# %%  Storage file to dataframe.
def mot_to_dataframe(mot_file, headers):
    # Extract data
    data = mot_to_numpy(mot_file)
    out = pd.DataFrame(data=data["time"], columns=["time"])
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])

    return out


# %% Load mot and output as dataframe or numpy
def load_mot(file_path, outputFormat="numpy"):
    table = opensim.TimeSeriesTable(file_path)
    data = table.getMatrix().to_numpy()
    time = np.asarray(table.getIndependentColumn()).reshape(-1, 1)
    data = np.hstack((time, data))
    headers = ["time"] + list(table.getColumnLabels())

    if outputFormat == "numpy":
        return data, headers
    elif outputFormat == "dataframe":
        return pd.DataFrame(data, columns=headers), headers
    else:
        return None

    # %%  Numpy array to mot file.


def numpy_to_mot(labels, data, mot_file, datatype=None):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"

    f = open(mot_file, "w")
    # Old style
    if datatype is None:
        f = open(mot_file, "w")
        f.write("name %s\n" % mot_file)
        f.write("datacolumns %d\n" % data.shape[1])
        f.write("datarows %d\n" % data.shape[0])
        f.write("range %f %f\n" % (np.min(data[:, 0]), np.max(data[:, 0])))
        f.write("endheader \n")
    # New style
    else:
        if datatype == "IK":
            f.write("Coordinates\n")
        elif datatype == "ID":
            f.write("Inverse Dynamics Generalized Forces\n")
        elif datatype == "GRF":
            f.write("%s\n" % mot_file)
        elif datatype == "muscle_forces":
            f.write("ModelForces\n")
        f.write("version=1\n")
        f.write("nRows=%d\n" % data.shape[0])
        f.write("nColumns=%d\n" % data.shape[1])
        if datatype == "IK":
            f.write("inDegrees=yes\n\n")
            f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
            f.write(
                "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n"
            )
        elif datatype == "ID":
            f.write("inDegrees=no\n")
        elif datatype == "GRF":
            f.write("inDegrees=yes\n")
        elif datatype == "muscle_forces":
            f.write("inDegrees=yes\n\n")
            f.write(
                "This file contains the forces exerted on a model during a simulation.\n\n"
            )
            f.write(
                "A force is a generalized force, meaning that it can be either a force (N) or a torque (Nm).\n\n"
            )
            f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
            f.write("Angles are in degrees.\n\n")

        f.write("endheader \n")

    for i in range(len(labels)):
        f.write("%s\t" % labels[i])
    f.write("\n")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write("%20.8f\t" % data[i, j])
        f.write("\n")

    f.close()
