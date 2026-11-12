def getOpenPoseMarkerNames():
    markerNames = [
        "Nose",
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
        "midHip",
        "RHip",
        "RKnee",
        "RAnkle",
        "LHip",
        "LKnee",
        "LAnkle",
        "REye",
        "LEye",
        "REar",
        "LEar",
        "LBigToe",
        "LSmallToe",
        "LHeel",
        "RBigToe",
        "RSmallToe",
        "RHeel",
    ]

    return markerNames


def getOpenPoseFaceMarkers():
    faceMarkerNames = ["Nose", "REye", "LEye", "REar", "LEar"]
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]

    return faceMarkerNames, idxFaceMarkers
