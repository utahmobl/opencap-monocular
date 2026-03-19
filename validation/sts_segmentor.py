import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd


# %%  Storage file to numpy array.
def storage_to_numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, "r")

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
    data = np.genfromtxt(storage_file, names=names, skip_header=skip_header)

    return data


def storage_to_dataframe(storage_file, headers):
    # Extract data
    data = storage_to_numpy(storage_file)
    out = pd.DataFrame(data=data["time"], columns=["time"])
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])

    return out


def filterNumpyArray(array, time, cutoff_frequency=6, order=4):

    fs = np.round(1 / np.mean(np.diff(time)), 6)
    fc = cutoff_frequency
    w = fc / (fs / 2)
    b, a = signal.butter(order / 2, w, "low")
    arrayFilt = signal.filtfilt(
        b, a, array, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1)
    )
    # print('numpy array filtered at {}Hz.'.format(cutoff_frequency))

    return arrayFilt


# %% Segment sit-to-stands.
"""
 Three time intervals are returned:
     - risingTimes: rising phase.
     - risingTimesDelayedStart: rising phase from delayed start to exclude
        time interval when there is contact with the chair.
     - risingSittingTimesDelayedStartPeriodicEnd: rising and sitting phases
         from delayed start to corresponding periodic end in terms of
         vertical pelvis position.     
"""


def segment_STS(
    ikFilePath,
    pelvis_ty=None,
    timeVec=None,
    velSeated=0.3,
    velStanding=0.15,
    visualize=False,
    filter_pelvis_ty=True,
    cutoff_frequency=4,
    delay=0.1,
):

    # Extract pelvis_ty if not given.
    if pelvis_ty is None and timeVec is None:
        ikResults = storage_to_dataframe(ikFilePath, headers={"pelvis_ty"})
        timeVec = ikResults["time"]
        if filter_pelvis_ty:
            pelvis_ty = filterNumpyArray(
                ikResults["pelvis_ty"].to_numpy(),
                timeVec.to_numpy(),
                cutoff_frequency=cutoff_frequency,
            )
        else:
            pelvis_ty = ikResults["pelvis_ty"]
    dt = timeVec[1] - timeVec[0]

    # Identify minimum.
    pelvSignal = np.array(pelvis_ty - np.min(pelvis_ty))
    pelvVel = np.diff(pelvSignal, append=0) / dt
    idxMaxPelvTy, _ = signal.find_peaks(
        pelvSignal, distance=0.9 / dt, height=0.2, prominence=0.2
    )

    # Find the max adjacent to all of the minimums.
    maxIdxOld = 0
    startFinishInds = []
    for i, maxIdx in enumerate(idxMaxPelvTy):
        # Find velocity peak to left of pelv_ty peak.
        vels = pelvVel[maxIdxOld:maxIdx]
        velPeak, peakVals = signal.find_peaks(vels, distance=0.9 / dt, height=0.2)
        velPeak = velPeak[np.argmax(peakVals["peak_heights"])] + maxIdxOld
        velsLeftOfPeak = np.flip(pelvVel[maxIdxOld:velPeak])
        velsRightOfPeak = pelvVel[velPeak:]
        # Trace left off the pelv_ty peak and find first index where
        # velocity<velSeated m/s.
        slowingIndLeft = np.argwhere(velsLeftOfPeak < velSeated)[0]
        startIdx = velPeak - slowingIndLeft
        slowingIndRight = np.argwhere(velsRightOfPeak < velStanding)[0]
        endIdx = velPeak + slowingIndRight
        startFinishInds.append([startIdx[0], endIdx[0]])
        maxIdxOld = np.copy(maxIdx)
    risingTimes = [timeVec[i].tolist() for i in startFinishInds]

    # We add a delay to make sure we do not simulate part of the motion
    # involving chair contact; this is not modeled.
    sf = 1 / np.round(
        np.mean(np.round(timeVec.to_numpy()[1:] - timeVec.to_numpy()[:-1], 2)), 16
    )
    startFinishIndsDelay = []
    for i in startFinishInds:
        c_i = []
        for c_j, j in enumerate(i):
            if c_j == 0:
                c_i.append(j + int(delay * sf))
            else:
                c_i.append(j)
        startFinishIndsDelay.append(c_i)
    risingTimesDelayedStart = [timeVec[i].tolist() for i in startFinishIndsDelay]

    # Segment periodic STS by identifying when the pelvis_ty value from the
    # standing phase best matches that from the sitting phase.
    startFinishIndsDelayPeriodic = []
    for val in startFinishIndsDelay:
        pelvVal_up = pelvSignal[val[0]]
        # Find next index when pelvis_ty is lower than this value.
        val_down = np.argwhere(pelvSignal[val[0] + 1 :] < pelvVal_up)[0][0]
        # Add trimmed part.
        val_down += val[0] + 1
        # Select val_down or val_down-1 based on best match with pelvVal_up.
        if np.abs(pelvSignal[val_down] - pelvVal_up) > np.abs(
            pelvSignal[val_down - 1] - pelvVal_up
        ):
            val_down -= 1
        startFinishIndsDelayPeriodic.append([val[0], val_down])
    risingSittingTimesDelayedStartPeriodicEnd = [
        timeVec[i].tolist() for i in startFinishIndsDelayPeriodic
    ]

    if visualize:
        plt.figure()
        # Plot against time instead of frame numbers
        plt.plot(timeVec, pelvSignal)
        for c_v, val in enumerate(startFinishInds):
            plt.plot(
                timeVec[val],
                pelvSignal[val],
                marker="o",
                markerfacecolor="k",
                markeredgecolor="none",
                linestyle="none",
                label="Rising phase",
            )
            val2 = startFinishIndsDelay[c_v][0]
            plt.plot(
                timeVec[val2],
                pelvSignal[val2],
                marker="o",
                markerfacecolor="r",
                markeredgecolor="none",
                linestyle="none",
                label="Delayed start",
            )
            val3 = startFinishIndsDelayPeriodic[c_v][1]
            plt.plot(
                timeVec[val3],
                pelvSignal[val3],
                marker="o",
                markerfacecolor="g",
                markeredgecolor="none",
                linestyle="none",
                label="Periodic end corresponding to delayed start",
            )
            if c_v == 0:
                plt.legend()
        # Update x-axis label to reflect time
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.title("Vertical pelvis position")
        plt.tight_layout()
        plt.show()

    return (
        risingTimes,
        risingTimesDelayedStart,
        risingSittingTimesDelayedStartPeriodicEnd,
    )


if __name__ == "__main__":
    # ikFilePath = "/home/selim/opencap-mono/output/case_001_STS/subject2/Session0/Cam3/STS1/OpenSim/IK/shiftedIK/STS1_5_sync_wham.mot"
    ikFilePath = "/home/selim/opencap-mono/output/case_001_STS/subject2/Session0/Cam3/STS1/OpenSim/IK/shiftedIK/STS1_5_sync.mot"

    risingTimes, risingTimesDelayedStart, risingSittingTimesDelayedStartPeriodicEnd = (
        segment_STS(ikFilePath)
    )
