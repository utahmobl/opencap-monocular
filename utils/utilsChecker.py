import sys

sys.path.append("./mmpose")  # utilities in child directory
import cv2
import numpy as np

import os
import glob
import shutil
import matplotlib.pyplot as plt
import scipy
from scipy.signal import gaussian
from scipy.signal import find_peaks
from utils.utilsCameraPy3 import Camera, nview_linear_triangulations
from itertools import combinations
import copy
from utils.utilsMarkers import getOpenPoseMarkerNames, getOpenPoseFaceMarkers


def getMissingMarkersCameras(keypoints2D):
    # Identify cameras that returned (0,0) as marker coordinates, ie that could
    # not identify the keypoints.
    # missingCams contains the indices of the cameras
    # missingMarkers contains the indices of the markers
    # Eg, missingCams[0] = 5 and missingMarkers[0] = 17 means that the camera
    # with index 5 returned (0,0) as coordinates of the marker with index 17.
    keypoints2D_res = np.reshape(
        np.stack(keypoints2D),
        (
            np.stack(keypoints2D).shape[0],
            np.stack(keypoints2D).shape[1],
            np.stack(keypoints2D).shape[3],
        ),
    )
    missingCams, missingMarkers = np.where(np.sum(keypoints2D_res, axis=2) == 0)

    return missingCams, missingMarkers


def triangulateMultiview(
    CameraParamList,
    points2dUndistorted,
    imageScaleFactor=1,
    useRotationEuler=False,
    ignoreMissingMarkers=False,
    selectCamerasMinReprojError=False,
    ransac=False,
    keypoints2D=[],
    confidence=None,
):
    # create a list of cameras (says sequence in documentation) from CameraParamList
    cameraList = []
    nCams = len(CameraParamList)
    nMkrs = np.shape(points2dUndistorted[0])[0]

    for camParams in CameraParamList:
        # get rotation matrix
        if useRotationEuler:
            rotMat = cv2.Rodrigues(camParams["rotation_EulerAngles"])[0]
        else:
            rotMat = camParams["rotation"]

        c = Camera()
        c.set_K(camParams["intrinsicMat"])
        c.set_R(rotMat)
        c.set_t(np.reshape(camParams["translation"], (3, 1)))
        cameraList.append(c)

    # triangulate
    stackedPoints = np.stack(points2dUndistorted)
    pointsInput = []
    for i in range(stackedPoints.shape[1]):
        pointsInput.append(stackedPoints[:, i, 0, :].T)

    points3d, confidence3d = nview_linear_triangulations(
        cameraList, pointsInput, weights=confidence
    )

    # Below are some outlier rejection methods

    # A slow, hacky way of rejecting outliers, like RANSAC.
    # Select the combination of cameras that minimize mean reprojection error for all cameras
    if selectCamerasMinReprojError and nCams > 2:

        # Function definitions
        def generateCameraCombos(nCams):
            comb = []
            k = nCams - 1
            comb.append(tuple(range(nCams)))
            while k >= 2:
                comb = comb + list(combinations(np.arange(nCams), k))
                k = k - 1
            return comb

        # generate a list of all possible camera combinations down to 2
        camCombos = generateCameraCombos(len(cameraList))

        # triangulate for all camera combiations
        points3DList = []
        nMkrs = confidence[0].shape[0]
        reprojError = np.empty((nMkrs, len(camCombos)))

        for iCombo, camCombo in enumerate(camCombos):
            camList = [cameraList[i] for i in camCombo]
            points2DList = [pts[:, camCombo] for pts in pointsInput]
            conf = [confidence[i] for i in camCombo]

            points3DList.append(
                nview_linear_triangulations(camList, points2DList, weights=conf)
            )

            # compute per-marker, confidence-weighted reprojection errors for all camera combinations
            # reprojError[:,iCombo] = calcReprojectionError(camList,points2DList,points3DList[-1],weights=conf)
            reprojError[:, iCombo] = calcReprojectionError(
                cameraList, pointsInput, points3DList[-1], weights=confidence
            )

        # select the triangulated point from camera set that minimized reprojection error
        new3Dpoints = np.empty((3, nMkrs))
        for iMkr in range(nMkrs):
            new3Dpoints[:, iMkr] = points3DList[np.argmin(reprojError[iMkr, :])][
                :, iMkr
            ]

        points3d = new3Dpoints

    # RANSAC for outlier rejection - on a per-marker basis, not a per-camera basis. Could be part of the problem
    # Not clear that this is helpful 4/23/21
    if ransac and nCams > 2:
        nIter = np.round(
            np.log(0.01) / np.log(1 - np.power(0.75, 2))
        )  # log(1 - prob of getting optimal set) / log(1-(n_inliers/n_points)^minPtsForModel)
        # TODO make this a function of image resolution
        errorUB = (
            20  # pixels, below this reprojection error, another camera gets added.
        )
        nGoodModel = 3  # model must have this many cameras to be considered good

        # functions
        def triangulateLimitedCameras(
            cameraList, pointsInput, confidence, cameraNumList
        ):
            camList = [cameraList[i] for i in cameraNumList]
            points2DList = [pts[:, cameraNumList] for pts in pointsInput]
            conf = [confidence[i] for i in cameraNumList]
            points3D = nview_linear_triangulations(camList, points2DList, weights=conf)
            return points3D

        def reprojErrorLimitedCameras(
            cameraList, pointsInput, points3D, confidence, cameraNumList
        ):
            if type(cameraNumList) is not list:
                cameraNumList = [cameraNumList]
            camList = [cameraList[i] for i in cameraNumList]
            points2DList = [pts[:, cameraNumList] for pts in pointsInput]
            conf = [confidence[i] for i in cameraNumList]
            reprojError = calcReprojectionError(
                camList, points2DList, points3D, weights=conf
            )
            return reprojError

        # initialize
        bestReprojError = 1000 * np.ones(nMkrs)  # initial, large value
        best3Dpoint = np.empty((points3d.shape))
        camComboList = [[] for _ in range(nMkrs)]

        for iIter in range(int(nIter)):
            np.random.seed(iIter)
            camCombo = np.arange(nCams)
            np.random.shuffle(
                camCombo
            )  # Seed setting should give same combos every run
            maybeInliers = list(camCombo[:2])
            alsoInliers = [[] for _ in range(nMkrs)]

            # triangulate maybe inliers
            points3D = triangulateLimitedCameras(
                cameraList, pointsInput, confidence, maybeInliers
            )

            # error on next camera
            for iMkr in range(nMkrs):
                for j in range(nCams - 2):
                    er = reprojErrorLimitedCameras(
                        cameraList, pointsInput, points3D, confidence, camCombo[2 + j]
                    )
                    # print(er[iMkr])
                    if er[iMkr] < errorUB:
                        alsoInliers[iMkr].append(camCombo[2 + j])
                # see if error is bigger than previous, if not, use this combo to triangulate
                # just 1 marker
                if (len(maybeInliers) + len(alsoInliers[iMkr])) >= nGoodModel:
                    thisConf = [np.atleast_1d(c[iMkr]) for c in confidence]
                    point3D = triangulateLimitedCameras(
                        cameraList,
                        [pointsInput[iMkr]],
                        thisConf,
                        maybeInliers + alsoInliers[iMkr],
                    )
                    er3D = reprojErrorLimitedCameras(
                        cameraList,
                        [pointsInput[iMkr]],
                        point3D,
                        thisConf,
                        maybeInliers + alsoInliers[iMkr],
                    )
                    # if er3D<bestReprojError[iMkr]:
                    if (len(maybeInliers) + len(alsoInliers[iMkr])) > len(
                        camComboList[iMkr]
                    ):
                        best3Dpoint[:, iMkr] = point3D.T
                        bestReprojError[iMkr] = er3D
                        camComboList[iMkr] = (
                            maybeInliers.copy() + alsoInliers[iMkr].copy()
                        )

        points3d = best3Dpoint

    if ignoreMissingMarkers and nCams > 2:
        # For markers that were not identified by certain cameras,
        # we re-compute their 3D positions but only using cameras that could
        # identify them (ie cameras that did not return (0,0) as coordinates).
        missingCams, missingMarkers = getMissingMarkersCameras(keypoints2D)

        for missingMarker in np.unique(missingMarkers):
            idx_missingMarker = np.where(missingMarkers == missingMarker)[0]
            idx_missingCam = missingCams[idx_missingMarker]

            idx_viewedCams = list(range(0, len(cameraList)))
            for i in idx_missingCam:
                idx_viewedCams.remove(i)

            CamParamList_viewed = [cameraList[i] for i in idx_viewedCams]
            c_pointsInput = copy.deepcopy(pointsInput)
            for count, pointInput in enumerate(c_pointsInput):
                c_pointsInput[count] = pointInput[:, idx_viewedCams]

            points3d_missingMarker = nview_linear_triangulations(
                CamParamList_viewed, c_pointsInput, weights=confidence
            )

            # overwritte marker
            points3d[:, missingMarker] = points3d_missingMarker[:, missingMarker]

    return points3d, confidence3d


def unpackKeypointList(keypointList):
    nFrames = keypointList[0].shape[1]
    unpackedKeypoints = []
    for iFrame in range(nFrames):
        tempList = []
        for keyArray in keypointList:
            tempList.append(keyArray[:, iFrame, None, :])
        unpackedKeypoints.append(tempList.copy())

    return unpackedKeypoints


def calcReprojectionError(
    cameraList, points2D, points3D, weights=None, normalizeError=False
):
    reprojError = np.empty((points3D.shape[1], len(cameraList)))

    if weights == None:
        weights = [1 for i in range(len(cameraList))]
    for iCam, cam in enumerate(cameraList):
        reproj = cam.world_to_image(points3D)[:2, :]
        this2D = np.array([pt2D[:, iCam] for pt2D in points2D]).T
        reprojError[:, iCam] = np.linalg.norm(
            np.multiply((reproj - this2D), weights[iCam]), axis=0
        )

        if normalizeError:  # Normalize by height of bounding box
            nonZeroYVals = this2D[1, :][this2D[1, :] > 0]
            boxHeight = np.nanmax(nonZeroYVals) - np.nanmin(nonZeroYVals)
            reprojError[:, iCam] /= boxHeight
    weightedReprojError_u = np.mean(reprojError, axis=1)
    return weightedReprojError_u


def triangulateMultiviewVideo(
    CameraParamDict,
    keypointDict,
    imageScaleFactor=1,
    ignoreMissingMarkers=False,
    keypoints2D=[],
    cams2Use=["all"],
    confidenceDict={},
    trimTrial=True,
    spline3dZeros=False,
    splineMaxFrames=5,
    nansInOut=[],
    CameraDirectories=None,
    trialName=None,
    startEndFrames=None,
    trialID="",
    outputMediaFolder=None,
):
    # cams2Use is a list of cameras that you want to use in triangulation.
    # if first entry of list is ['all'], will use all
    # otherwise, ['Cam0','Cam2']
    CameraParamList = [CameraParamDict[i] for i in CameraParamDict]
    if cams2Use[0] == "all" and not None in CameraParamList:
        keypointDict_selectedCams = keypointDict
        CameraParamDict_selectedCams = CameraParamDict
        confidenceDict_selectedCams = confidenceDict
    else:
        if cams2Use[0] == "all":  # must have been a none (uncalibrated camera)
            cams2Use = []
            for camName in CameraParamDict:
                if CameraParamDict[camName] is not None:
                    cams2Use.append(camName)

        keypointDict_selectedCams = {}
        CameraParamDict_selectedCams = {}
        if confidenceDict:
            confidenceDict_selectedCams = {}
        for camName in cams2Use:
            if CameraParamDict[camName] is not None:
                keypointDict_selectedCams[camName] = keypointDict[camName]
                CameraParamDict_selectedCams[camName] = CameraParamDict[camName]
                if confidenceDict:
                    confidenceDict_selectedCams[camName] = confidenceDict[camName]

    keypointList_selectedCams = [
        keypointDict_selectedCams[i] for i in keypointDict_selectedCams
    ]
    confidenceList_selectedCams = [
        confidenceDict_selectedCams[i] for i in confidenceDict_selectedCams
    ]
    CameraParamList_selectedCams = [
        CameraParamDict_selectedCams[i] for i in CameraParamDict_selectedCams
    ]
    unpackedKeypoints = unpackKeypointList(keypointList_selectedCams)
    points3D = np.zeros(
        (
            3,
            keypointList_selectedCams[0].shape[0],
            keypointList_selectedCams[0].shape[1],
        )
    )
    confidence3D = np.zeros(
        (
            1,
            keypointList_selectedCams[0].shape[0],
            keypointList_selectedCams[0].shape[1],
        )
    )

    for iFrame, points2d in enumerate(unpackedKeypoints):
        # If confidence weighting
        if confidenceDict:
            thisConfidence = [c[:, iFrame] for c in confidenceList_selectedCams]
        else:
            thisConfidence = None

        points3D[:, :, iFrame], confidence3D[:, :, iFrame] = triangulateMultiview(
            CameraParamList_selectedCams,
            points2d,
            imageScaleFactor=1,
            useRotationEuler=False,
            ignoreMissingMarkers=ignoreMissingMarkers,
            keypoints2D=keypoints2D,
            confidence=thisConfidence,
        )

    if trimTrial:
        # Delete confidence and 3D keypoints if markers, except for face
        # markers, have 0 confidence (they're garbage b/c <2 cameras saw them).
        markerNames = getOpenPoseMarkerNames()
        allMkrInds = np.arange(len(markerNames))
        _, idxFaceMarkers = getOpenPoseFaceMarkers()
        includedMkrs = np.delete(allMkrInds, idxFaceMarkers)

        nZeroConf = len(includedMkrs) - np.count_nonzero(
            confidence3D[:, includedMkrs, :], axis=1
        )
        if not True in (nZeroConf < 1).flatten():
            points3D = np.zeros((3, 25, 10))
            confidence3D = np.zeros((1, 25, 10))
            startInd = 0
            endInd = confidence3D.shape[2]

        else:
            startInd = np.argwhere(nZeroConf < 1)[0, 1]
            endInd = confidence3D.shape[2] - np.argwhere(np.flip(nZeroConf < 1))[0, 1]

            # If there were less than 3 cameras, then we also take into account the
            # inleading and exiting nans, which result in garbage interpolated
            # keypoints.
            if nansInOut:
                nans_in = [nansInOut[cam][0] for cam in nansInOut]
                nans_out = [nansInOut[cam][1] for cam in nansInOut]
                # When >2 cameras, all list entries will be nan. If >2 cameras,
                # but only 2 see person, will have 2 non-nan entries.
                if not any(np.isnan(nans_in)) or np.sum(~np.isnan(nans_in)) == 2:
                    startInd = int(
                        np.nanmax(np.array([startInd, np.nanmax(np.asarray(nans_in))]))
                    )
                    endInd = int(
                        np.nanmin(np.array([endInd, np.nanmin(np.asarray(nans_out))]))
                    )

            # nPointsOriginal = copy.deepcopy(points3D.shape[2])
            points3D = points3D[:, :, startInd:endInd]
            confidence3D = confidence3D[:, :, startInd:endInd]
    else:
        startInd = 0
        endInd = confidence3D.shape[2]

    # Rewrite videos based on sync time and trimmed trc.
    if CameraDirectories != None and trialName != None:
        print("Writing synchronized videos")
        outputVideoDir = os.path.abspath(
            os.path.join(
                list(CameraDirectories.values())[0],
                "../../",
                "VisualizerVideos",
                trialName,
            )
        )
        # Check if the directory already exists
        if os.path.exists(outputVideoDir):
            # If it exists, delete it and its contents
            shutil.rmtree(outputVideoDir)
        os.makedirs(outputVideoDir, exist_ok=True)
        for iCam, camName in enumerate(keypointDict):

            nFramesToWrite = endInd - startInd

            if outputMediaFolder is None:
                outputMediaFolder = "OutputMedia*"

            inputPaths = glob.glob(
                os.path.join(
                    CameraDirectories[camName],
                    outputMediaFolder,
                    trialName,
                    trialID + "*",
                )
            )
            if len(inputPaths) > 0:
                inputPath = inputPaths[0]
            else:
                inputPaths = glob.glob(
                    os.path.join(
                        CameraDirectories[camName],
                        "InputMedia*",
                        trialName,
                        trialID + "*",
                    )
                )
                inputPath = inputPaths[0]

            # get frame rate and assume all the same for sync'd videos
            if iCam == 0:
                thisVideo = cv2.VideoCapture(inputPath.replace(".mov", "_rotated.avi"))
                frameRate = np.round(thisVideo.get(cv2.CAP_PROP_FPS))
                thisVideo.release()

            # Only rewrite if camera in cams2use and wasn't kicked out earlier
            if (camName in cams2Use or cams2Use[0] == "all") and startEndFrames[
                camName
            ] != None:
                _, inputName = os.path.split(inputPath)
                inputRoot, inputExt = os.path.splitext(inputName)

                # Let's use mp4 since we write for the internet
                outputFileName = inputRoot + "_syncd_" + camName + ".mp4 "  # inputExt

                thisStartFrame = startInd + startEndFrames[camName][0]

                # rewriteVideos(inputPath, thisStartFrame, nFramesToWrite, frameRate,
                #               outputDir=outputVideoDir, imageScaleFactor=.5,
                #               outputFileName=outputFileName)

    return points3D, confidence3D


def calcReprojectionErrorForSync(
    CamParamList, keypointList, lagVal, cams2UseReproj, confidence, cameras2Use
):
    # Number of timesteps to triangulate. Will average reprojection error over all nTimesteps.
    nTimesteps = 5

    keypoints2D = copy.deepcopy(keypointList)
    conf = copy.deepcopy(confidence)
    CamParamListCopy = copy.deepcopy(CamParamList)

    # Find the range of overlapping confidence for this lag value
    confSel = []
    for cam in cams2UseReproj:
        confSel.append(conf[cam])

    # find confidence ranges in original indices
    confThresh = [
        0.5 * np.nanmax(c) for c in confSel
    ]  # Threshold for saying this camera confidently sees the person
    avgConf = [np.nanmean(c, axis=0) for c in confSel]

    # Breakdown steps to catch potential error.
    confRanges = []
    for i, c in enumerate(avgConf):
        temp = c > confThresh[i]
        if True in temp:
            confRanges.append(
                np.array([np.argwhere(temp)[0], np.argwhere(temp)[-1] + 1])
            )
        else:
            reprojErrorAcrossFrames = 0.1
            reprojSuccess = False
            return reprojErrorAcrossFrames, reprojSuccess

    # shift second camera based on lag, so indices are "aligned," then find overlapping range
    shiftedConfRanges = copy.deepcopy(confRanges)
    shiftedConfRanges[1] = (
        shiftedConfRanges[1] - lagVal
    )  # shift the indices for second camera
    # Ignore the first and last few timesteps here as confidence drops
    shiftedOverlapInds = [
        np.max((shiftedConfRanges[0][0], shiftedConfRanges[1][0])) + 3,
        np.min((shiftedConfRanges[0][1], shiftedConfRanges[1][1])) - 3,
    ]

    # Sample nTimesteps between the shifted Overlap Inds
    shiftedSampleInds = np.linspace(
        shiftedOverlapInds[0], shiftedOverlapInds[1], nTimesteps
    ).astype(int)

    sampleInds = []
    sampleInds.append(shiftedSampleInds)  # no shift for first camera
    sampleInds.append(
        shiftedSampleInds + lagVal
    )  # unshifts the indices for second camera

    # Select keypoints and confidence at appropriate timesteps
    keypoints2DSelected = []
    cameraListSelected = []
    confListSelected = []
    for iCam, cam in enumerate(cams2UseReproj):
        keypoints2D[cam] = keypoints2D[cam][:, sampleInds[iCam], :]
        conf[cam] = conf[cam][:, sampleInds[iCam]]
        keypoints2DSelected.append(keypoints2D[cam])
        cameraListSelected.append(CamParamListCopy[cam])
        confListSelected.append(conf[cam])

    # Triangulate at each of the nTimesteps
    # We here need to turn the lists back into dicts.
    CamParamListCopy_dict = {}
    keypoints2D_dict = {}
    conf_dict = {}
    for iCam, cam in enumerate(cameras2Use):
        CamParamListCopy_dict[cam] = CamParamListCopy[iCam]
        keypoints2D_dict[cam] = keypoints2D[iCam]
        conf_dict[cam] = conf[iCam]
    cameras2UseReproj = [cameras2Use[i] for i in cams2UseReproj]
    keypoints3D, _ = triangulateMultiviewVideo(
        CamParamListCopy_dict,
        keypoints2D_dict,
        ignoreMissingMarkers=False,
        cams2Use=cameras2UseReproj,
        confidenceDict=conf_dict,
        trimTrial=False,
    )

    # Make list of camera objects
    cameraObjList = []
    for camParams in cameraListSelected:
        c = Camera()
        c.set_K(camParams["intrinsicMat"])
        c.set_R(camParams["rotation"])
        c.set_t(np.reshape(camParams["translation"], (3, 1)))
        cameraObjList.append(c)

    # Compute confidence-weighted reprojection error for each of nTimesteps
    reprojErrorVec = []
    for tStep in range(nTimesteps):

        # Organize points for reprojectionError function
        stackedPoints = np.stack([k[:, None, tStep, :] for k in keypoints2DSelected])
        pointsInput = []
        for i in range(stackedPoints.shape[1]):
            pointsInput.append(stackedPoints[:, i, 0, :].T)

        confForWeights = [
            c[:, None, tStep].T for c in confListSelected
        ]  # needs transpose for reprojection error function
        confForWeights = [
            np.nan_to_num(c, nan=0) for c in confForWeights
        ]  # sometimes confidence has nans, don't want to use as weights in this case

        # Calculate combined reprojection error
        key3D = np.squeeze(keypoints3D[:, :, tStep])
        reprojErrors = calcReprojectionError(
            cameraObjList,
            pointsInput,
            key3D,
            weights=confForWeights,
            normalizeError=True,
        )

        # multiply minimum confidence between cameras times marker-wise reproj errors
        # so we don't include errors for markers that had low confidence in one of the cameras
        minConfVec = np.min(np.asarray(confForWeights), axis=0)
        minConfVec[np.where(minConfVec < 0.5)] = 0  # Set low conf markers to 0
        weightedReprojErrors = np.multiply(reprojErrors, minConfVec)
        if np.any(weightedReprojErrors):
            reprojErrorVec.append(np.mean(weightedReprojErrors[minConfVec > 0]))
        else:
            reprojErrorVec.append(
                1000
            )  # in cases where no position is confident set to large reproj error. typical values are on the order of  0.1

    reprojErrorAcrossFrames = np.mean(reprojErrorVec)
    reprojSuccess = True

    return reprojErrorAcrossFrames, reprojSuccess


def cross_corr_multiple_timeseries(
    Y1,
    Y2,
    multCorrGaussianStd=None,
    dataForReproj=None,
    visualize=False,
    frameRate=60,
    path=None,
    approximateLag=None,
):
    """
    GPT generated docstring.
    Calculates the cross-correlation and lag for multiple timeseries without normalization.

    This function computes the cross-correlation between two sets of time-series data (Y1 and Y2),
    optionally applies Gaussian weighting to the correlation, and can refine the lag estimation using reprojection error.

    Args:
        Y1 (np.ndarray): A 2D array of shape (nMkrs, nSamples) representing the first set of time-series data.
        Y2 (np.ndarray): A 2D array of shape (nMkrs, nSamples) representing the second set of time-series data.
        multCorrGaussianStd (float, optional): Standard deviation for a Gaussian to prioritize correlation peaks near zero lag.
            If None, no Gaussian weighting is applied.
        dataForReproj (dict, optional): Data for reprojection error minimization. Should include:
            - 'CamParamList': List of camera parameters.
            - 'keypointList': List of detected keypoints.
            - 'cams2UseReproj': Cameras to use for reprojection.
            - 'confidence': Confidence scores for keypoints.
            - 'cameras2Use': Cameras being used for synchronization.
        visualize (bool, optional): Whether to plot intermediate results (default is False).
        frameRate (int, optional): Frame rate of the timeseries data (default is 60).
        path (str, optional): Path to save visualizations. Must end with a '/'.

    Returns:
        max_corr (float): Maximum correlation value.
        lag (int): The lag (in frames) corresponding to the maximum correlation.

    Notes:
        - The definition of cross-correlation is consistent with the MATLAB `xcorr` function.
        - If `dataForReproj` is provided, the lag is refined based on reprojection error minimization.
        - If `multCorrGaussianStd` is provided, a Gaussian weighting is applied to prioritize lag solutions near zero.

    """
    nMkrs = Y1.shape[0]
    corrMat = np.empty(Y1.shape)
    for iMkr in range(nMkrs):
        y1 = Y1[iMkr, :]
        y2 = Y2[iMkr, :]
        # Pad shorter signal with 0s
        if len(y1) > len(y2):
            temp = np.zeros(len(y1))
            temp[0 : len(y2)] = y2
            y2 = np.copy(temp)
        elif len(y2) > len(y1):
            temp = np.zeros(len(y2))
            temp[0 : len(y1)] = y1
            y1 = np.copy(temp)

        y1_auto_corr = np.dot(y1, y1) / len(y1)
        y2_auto_corr = np.dot(y2, y2) / len(y1)
        corr = np.correlate(y1, y2, mode="same")
        # The unbiased sample size is N - lag
        unbiased_sample_size = np.correlate(
            np.ones(len(y1)), np.ones(len(y1)), mode="same"
        )
        corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
        shift = len(y1) // 2
        corrMat[iMkr, :] = corr

    if visualize:
        plt.figure()
        plt.plot(corrMat.T)
        plt.plot(np.nansum(corrMat, axis=0), color="k")
        plt.title("multi-marker correlation")
        legNames = ["mkr" + str(iMkr) for iMkr in range(nMkrs)]
        legNames.append("summedCorrelation")
        plt.legend(legNames)
        # save the plot
        if path is not None:
            plt.savefig(os.path.join(path, "cross_corr_multiple_timeseries.png"))

        plt.figure()
        plt.plot(Y1.T)
        plt.plot(Y2.T)

    summedCorr = np.nansum(corrMat, axis=0)

    # find correlation peak with minimum reprojection error
    if dataForReproj is not None:
        _, peaks = find_peaks(summedCorr, height=0.75)
        idxPeaks = np.squeeze(
            np.asarray(
                [
                    np.argwhere(peaks["peak_heights"][i] == summedCorr)
                    for i in range(len(peaks["peak_heights"]))
                ]
            )
        )
        lags = idxPeaks - shift
        # look at 3 lags closest to 0
        if np.isscalar(lags):
            max_corr = summedCorr[lags + shift]
            return max_corr, lags
        if len(lags) > 3:
            lags = lags[np.argsort(np.abs(lags))[:3]]

        reprojError = np.empty((len(lags), 1))
        reprojSuccess = []
        for iPeak, lag in enumerate(lags):
            # calculate reprojection error for each potential lag
            reprojError[iPeak, 0], reprojSuccessi = calcReprojectionErrorForSync(
                dataForReproj["CamParamList"],
                dataForReproj["keypointList"],
                lag,
                dataForReproj["cams2UseReproj"],
                dataForReproj["confidence"],
                dataForReproj["cameras2Use"],
            )
            reprojSuccess.append(reprojSuccessi)

        # find if min reproj error is clearly smaller than other peaks. If it is not,
        # don't use reproj error min for sync. E.g. with treadmill walking, reproj error may not work as
        # robustly as for overground walking
        reprojSort = np.sort(reprojError, axis=0)
        reprojErrorRatio = reprojSort[0] / reprojSort[1]
        if (
            reprojErrorRatio < 0.6 and not False in reprojSuccess
        ):  # tunable parameter. Usually around 0.25 for overground walking
            # find idx with minimum reprojection error
            lag_corr = lags[np.argmin(reprojError)]
            max_corr = summedCorr[lag + shift]

            if multCorrGaussianStd is not None:
                print(
                    "For {}, used reprojection error minimization to sync.".format(
                        dataForReproj["cameras2Use"][dataForReproj["cams2UseReproj"][1]]
                    )
                )

            # Refine with reprojection error minimization
            # Calculate reprojection error with lags +/- .2 seconds around the selected lag. Select the lag with the lowest reprojection error.
            # This helps the fact that correlation peak is not always the best lag, esp for front-facing cameras

            # Create a list of lags to test that is +/- .2 seconds around the selected lag based on frameRate
            numFrames = int(0.2 * frameRate)
            lags = np.arange(lag_corr - numFrames, lag_corr + numFrames + 1)
            reprojErrors = np.empty((len(lags), 1))

            for iLag, lag in enumerate(lags):
                reprojErrors[iLag, 0], _ = calcReprojectionErrorForSync(
                    dataForReproj["CamParamList"],
                    dataForReproj["keypointList"],
                    lag,
                    dataForReproj["cams2UseReproj"],
                    dataForReproj["confidence"],
                    dataForReproj["cameras2Use"],
                )

            # Select the lag with the lowest reprojection error
            lag = lags[np.argmin(reprojErrors)]

            # plot the reproj errors against lag and identify which was lag_corr
            if visualize:
                plt.figure()
                plt.plot(lags, reprojErrors)
                plt.plot(
                    lag_corr,
                    reprojErrors[list(lags).index(lag_corr)],
                    marker="o",
                    color="r",
                )
                plt.plot(
                    lag, reprojErrors[list(lags).index(lag)], marker="o", color="k"
                )
                plt.xlabel("lag")
                plt.ylabel("reprojection error")
                plt.title("Reprojection error vs lag")
                plt.legend(["reprojection error", "corr lag", "refined lag"])
                plt.show()

            return max_corr, lag

    # Multiply correlation curve by gaussian (prioritizing lag solution closest to 0)
    if multCorrGaussianStd is not None:
        summedCorr = np.multiply(
            summedCorr, gaussian(len(summedCorr), multCorrGaussianStd)
        )
        if visualize:
            plt.plot(summedCorr, color=(0.4, 0.4, 0.4))
            if path is not None:
                plt.savefig(
                    os.path.join(
                        path, "cross_corr_multiple_timeseries_multCorrGaussianStd.png"
                    )
                )

    _, peaks = find_peaks(summedCorr, height=0.75)
    idxPeaks = np.squeeze(
        np.asarray(
            [
                np.argwhere(peaks["peak_heights"][i] == summedCorr)
                for i in range(len(peaks["peak_heights"]))
            ]
        )
    )
    lags = idxPeaks - shift
    # look at 3 lags closest to 0
    # print("Lags: ", lags)
    # print('type(lags): ', type(lags))
    # print("summedCorr[lags + shift]", summedCorr[lags + shift])
    if not isinstance(lags, np.ndarray):
        return summedCorr[lags + shift], lags

    if approximateLag is not None:
        # sort the lags by the difference between the approximate lag and the lag
        if isinstance(lags, np.ndarray):
            sorted_lags = lags[np.argsort(np.abs(lags - approximateLag))]
            return summedCorr[sorted_lags[0] + shift], sorted_lags[0]

    # if len(lags) > 3:
    #     lags = lags[np.argsort(np.abs(lags))[:3]]
    #     # take the lag with the highest peak
    #     lags = lags[np.argsort(peaks['peak_heights'])[::-1]]
    #     max_corr = summedCorr[lags[0] + shift]
    #     return max_corr, lags[0]


def cross_corr(
    y1, y2, multCorrGaussianStd=None, visualize=False, dataForReproj=None, frameRate=60
):
    """Calculates the cross correlation and lags without normalization.

    The definition of the discrete cross-correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html

    Args:
    y1, y2: Should have the same length.

    Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
    """
    # Pad shorter signal with 0s
    if len(y1) > len(y2):
        temp = np.zeros(len(y1))
        temp[0 : len(y2)] = y2
        y2 = np.copy(temp)
    elif len(y2) > len(y1):
        temp = np.zeros(len(y2))
        temp[0 : len(y1)] = y1
        y1 = np.copy(temp)

    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode="same")
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(np.ones(len(y1)), np.ones(len(y1)), mode="same")
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2
    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)

    # find correlation peak with minimum reprojection error
    if dataForReproj is not None:
        _, peaks = find_peaks(corr, height=0.1)

        # inject no delay so it doesn't throw an error for static trials
        if len(peaks["peak_heights"]) == 0:
            peaks["peak_heights"] = np.ndarray((1, 1))
            peaks["peak_heights"][0] = corr[int(len(corr) / 2)]
            print("There were no peaks in the vert vel cross correlation. Using 0 lag.")
        idxPeaks = np.squeeze(
            np.asarray(
                [
                    np.argwhere(peaks["peak_heights"][i] == corr)
                    for i in range(len(peaks["peak_heights"]))
                ]
            )
        )
        lags = idxPeaks - shift
        # look at 3 lags closest to 0
        if np.isscalar(lags):
            max_corr = corr[lags + shift]
            return max_corr, lags
        if len(lags) > 3:
            lags = lags[np.argsort(np.abs(lags))[:3]]

        reprojError = np.empty((len(lags), 1))
        reprojSuccess = []
        for iPeak, lag in enumerate(lags):
            # calculate reprojection error for each potential lag
            reprojError[iPeak, 0], reprojSuccessi = calcReprojectionErrorForSync(
                dataForReproj["CamParamList"],
                dataForReproj["keypointList"],
                lag,
                dataForReproj["cams2UseReproj"],
                dataForReproj["confidence"],
                dataForReproj["cameras2Use"],
            )
            reprojSuccess.append(reprojSuccessi)

        # find if min reproj error is clearly smaller than other peaks. If it is not,
        # don't use reproj error min for sync. E.g. with treadmill walking, reproj error may not work as
        # robustly as for overground walking
        reprojSort = np.sort(reprojError, axis=0)
        reprojErrorRatio = reprojSort[0] / reprojSort[1]
        if (
            reprojErrorRatio < 0.6 and not False in reprojSuccess
        ):  # tunable parameter. Usually around 0.25 for overground walking
            # find idx with minimum reprojection error
            lag_corr = lags[np.argmin(reprojError)]
            max_corr = corr[lag + shift]

            if multCorrGaussianStd is not None:
                print(
                    "For {}, used reprojection error minimization to sync.".format(
                        dataForReproj["cameras2Use"][dataForReproj["cams2UseReproj"][1]]
                    )
                )

            # Refine with reprojection error minimization
            # Calculate reprojection error with lags +/- .2 seconds around the selected lag. Select the lag with the lowest reprojection error.
            # This helps the fact that correlation peak is not always the best lag, esp for front-facing cameras

            # Create a list of lags to test that is +/- .2 seconds around the selected lag based on frameRate
            numFrames = int(0.2 * frameRate)
            lags = np.arange(lag_corr - numFrames, lag_corr + numFrames + 1)
            reprojErrors = np.empty((len(lags), 1))

            for iLag, lag in enumerate(lags):
                reprojErrors[iLag, 0], _ = calcReprojectionErrorForSync(
                    dataForReproj["CamParamList"],
                    dataForReproj["keypointList"],
                    lag,
                    dataForReproj["cams2UseReproj"],
                    dataForReproj["confidence"],
                    dataForReproj["cameras2Use"],
                )

            # Select the lag with the lowest reprojection error
            lag = lags[np.argmin(reprojErrors)]

            # plot the reproj errors against lag and identify which was lag_corr
            if visualize:
                plt.figure()
                plt.plot(lags, reprojErrors)
                plt.plot(
                    lag_corr,
                    reprojErrors[list(lags).index(lag_corr)],
                    marker="o",
                    color="r",
                )
                plt.plot(
                    lag, reprojErrors[list(lags).index(lag)], marker="o", color="k"
                )
                plt.xlabel("lag")
                plt.ylabel("reprojection error")
                plt.title("Reprojection error vs lag")
                plt.legend(["reprojection error", "corr lag", "refined lag"])
                plt.show()

            return max_corr, lag

    if visualize:
        plt.figure()
        plt.plot(corr)
        plt.title("vertical velocity correlation")

    # Multiply correlation curve by gaussian (prioritizing lag solution closest to 0)
    if multCorrGaussianStd is not None:
        corr = np.multiply(corr, gaussian(len(corr), multCorrGaussianStd))
        if visualize:
            plt.plot(corr, color=[0.4, 0.4, 0.4])
            plt.legend(["corr", "corr*gaussian"])

    argmax_corr = np.argmax(corr)
    max_corr = np.nanmax(corr)

    lag = argmax_corr - shift

    return max_corr, lag


def detectFeetMoving(allMarkers, confidence, ankleInds, motionThreshold=0.5):
    # motion threshold is a percent of bounding box height/width

    # Get bounding box height or width
    # nFrames x(nMkrsx3)
    nFrames = confidence.shape[1]
    nMkrs = confidence.shape[0]
    cMkrs = np.copy(allMarkers)
    reshapedMkrs = np.ndarray((nFrames, 0))
    for i in range(nMkrs):
        reshapedMkrs = np.append(reshapedMkrs, np.squeeze(cMkrs[i, :, :]), axis=1)

    inData = np.insert(
        reshapedMkrs.T, np.arange(2, nMkrs * 2 + 1, 2), confidence, axis=0
    )

    bbox = keypointsToBoundingBox(inData.T)
    # normalize by the average width of the bounding box
    normValue = np.mean(bbox[:, 2])

    # compute max distance
    ankleMkrs = np.divide(allMarkers[ankleInds, :, :], normValue)
    ankleConf = confidence[ankleInds, :]
    confThresh = 0.4
    maxMvt = []
    for i in range(2):
        confidentInds = ankleConf[i] > confThresh
        if len(confidentInds) > 0:
            confidentMarkers = ankleMkrs[i, ankleConf[i] > confThresh, :]
            # need to find the two points that are furthest from each other. A naive
            # search is O(n^2). Let's assume we are looking for motion in horizontal
            # direction.
            idxMax = np.argmax(confidentMarkers[:, 0])
            idxMin = np.argmin(confidentMarkers[:, 0])
            maxMvt.append(
                scipy.linalg.norm(
                    confidentMarkers[idxMax, :] - confidentMarkers[idxMin, :]
                )
            )
        else:
            # if we did not see the foot, assume it did not move
            maxMvt.append = 0

    # did both feet move greater than the motion threshold
    anyFootMoving = all([m > motionThreshold for m in maxMvt])

    return anyFootMoving


def detectGait(rSpeed, lSpeed, frameRate):

    # cross correlate to see if they are in or out of phase
    corr, lag = cross_corr(rSpeed, lSpeed, multCorrGaussianStd=frameRate * 3)

    # default false in case feet are static
    isGait = False
    if corr > 0.55:
        if np.abs(lag) > 0.1 * frameRate and np.abs(lag) < frameRate:
            isGait = True

    return isGait


def detectGaitAllVideos(mkrSpeedList, allMarkers, confidence, ankleInds, sampleFreq):
    isGaits = []
    feetMoving = []
    for c_mkrSpeed, allMkrs, conf in zip(mkrSpeedList, allMarkers, confidence):
        isGaits.append(detectGait(c_mkrSpeed[0], c_mkrSpeed[1], sampleFreq))
        feetMoving.append(detectFeetMoving(allMkrs, conf, ankleInds))
    if len(isGaits) > 2:
        true_count = sum(isGaits)
        isGait = true_count >= len(isGaits) - 1
    else:
        isGait = all(isGaits)

    isGait = isGait and any(feetMoving)

    return isGait
