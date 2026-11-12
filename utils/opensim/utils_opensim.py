#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:22:03 2024

@author: Scott Uhlrich

Many functions copied from opencap-core repository
"""

import sys

import opensim
import numpy as np
import os
import json
import torch
import pickle
from loguru import logger
import utils.utils_optim as utils_optim
import utils.utils_trc as utils_trc
from utils.opensim.defaults import defaults
from utils.data.constants import AUGMENTED_VERTICES_INDEX_DICT, OPENPOSE_VERTICES_NAME
import pandas as pd

from slahmr.slahmr.util.loaders import load_smpl_body_model

# SMPL is usually X left, Y up, Z forward; opensim is X forward, Y up, Z right
default_rotations = {"x": 0, "y": 90, "z": 0}


class SMPL_to_Opensim:

    def __init__(
        self,
        smpl_model,
        root_trans,
        root_orient,
        body_pose,
        beta,
        output_dir,
        trial_name,
        frame_rate,
        mass=None,
    ):
        """
        Params

        smpl_model: SMPL Model
        root_trans: T,3 tensor
        root_orient: T,3 tensor in axis-angle
        body_pose: T,nJoints*3  tensor (69 for SMPL) in axis angle
        beta: T,nBeta or nBeta tensor
        output_dir
        trial_name
        frame_rate: FPS int
        mass : mass in kg, can leave blank if just want IK, but will give inaccurate dynamics
        opensim_model_path = None
        """

        self.smpl_model = smpl_model
        self.root_trans = root_trans.unsqueeze(0)  # expects B,T,3 below
        self.root_orient = root_orient.unsqueeze(0)
        self.body_pose = body_pose.unsqueeze(0)
        self.beta = beta

        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.trial_name = trial_name

        if mass is not None:  # Can get through IK without needing mass
            self.mass = mass
        else:
            print("using default mass of 75kg...will not give accurate dynamics.")
            mass = 75  # default Rajagopal mass

        # Create folders
        self.osim_dir = os.path.join(output_dir, "OpenSim")
        os.makedirs(self.osim_dir, exist_ok=True)

        # Generic paths to Opensim model and setup files

        self.opensim_model_path_generic = defaults()["OPENSIM_MODEL_PATH"]
        self.opensim_scaling_setup_path_generic = defaults()[
            "OPENSIM_SCALING_SETUP_PATH"
        ]
        self.opensim_ik_setup_path_generic = defaults()["OPENSIM_IK_SETUP_PATH"]

    def scale_model(self, rotations=default_rotations, opensim_model_path_generic=None):

        if opensim_model_path_generic is not None:
            self.opensim_model_path_generic = opensim_model_path_generic

        # logger.info(f"Using model: {self.opensim_model_path_generic}")
        framerate = 30  # this is a statically posed model, so it doesn't matter if it matches real video framerate
        length_trial = 1
        n_frames = length_trial * framerate

        self.osim_model_dir = os.path.join(self.osim_dir, "Model", self.trial_name)
        os.makedirs(self.osim_model_dir, exist_ok=True)

        # Convert neutral posed smpl to smplx

        device = self.beta.device
        trans_0 = torch.zeros(1, n_frames, 3, dtype=torch.float32, device=device)
        root_orient_0 = torch.zeros(1, n_frames, 3, dtype=torch.float32, device=device)
        body_pose_0 = torch.zeros(1, n_frames, 69, dtype=torch.float32, device=device)
        # If you want an A-pose for scaling. Shoulders are tough, but we got one to work with T-pose
        # body_pose_0[0,16*3:16*3+3] = torch.tensor([.186, .234, 1.17]) # Rshoulder in A pose
        # body_pose_0[0,15*3:15*3+3] = torch.tensor([.186, -.234, -1.17]) # Lshoulder in A pose

        smplx_vertices, openpose_markers = smpl_to_smplx_verts(
            root_orient_0, trans_0, body_pose_0, self.beta
        )

        marker_names, marker_positions = smplx_to_markers(
            smplx_vertices, openpose_markers
        )
        marker_positions = marker_positions.reshape(marker_positions.shape[0], -1)

        neutral_path = os.path.join(self.osim_model_dir, "neutral.trc")
        # logger.info(f"Writing neutral trc to {neutral_path}")

        utils_trc.write_trc(
            marker_positions,
            neutral_path,
            marker_names,
            frameRate=framerate,
            rotationAngles={
                "X": rotations["x"],
                "Y": rotations["y"],
                "Z": rotations["z"],
            },
        )

        # logger.info(f"Using scaling setup: {self.opensim_scaling_setup_path_generic}")

        # Scale osim model
        self.osim_scaled_model_path = runScaleTool(
            pathGenericSetupFile=self.opensim_scaling_setup_path_generic,
            pathGenericModel=self.opensim_model_path_generic,
            subjectMass=self.mass,
            pathTRCFile=neutral_path,
            timeRange=[0, length_trial],
            pathOutputFolder=self.osim_model_dir,
        )

        # logger.info(f"Model scaled to {self.osim_scaled_model_path}")

        return self.osim_scaled_model_path

    def smpl_to_trc(self, rotations=default_rotations):

        self.trc_dir = os.path.join(self.output_dir, "MarkerData", self.trial_name)
        os.makedirs(self.trc_dir, exist_ok=True)

        smplx_vertices, openpose_markers = smpl_to_smplx_verts(
            self.root_orient, self.root_trans, self.body_pose, self.beta
        )

        marker_names, marker_positions = smplx_to_markers(
            smplx_vertices, openpose_markers
        )
        marker_positions = marker_positions.reshape(marker_positions.shape[0], -1)

        # find the minimum y position
        min_y = np.min(marker_positions[:, 1::3])

        self.trc_path = os.path.join(self.trc_dir, self.trial_name + ".trc")

        utils_trc.write_trc(
            marker_positions,
            self.trc_path,
            marker_names,
            frameRate=self.frame_rate,
            rotationAngles={
                "X": rotations["x"],
                "Y": rotations["y"],
                "Z": rotations["z"],
            },
            vertical_offset=min_y,
        )

        return self.trc_path

    def run_ik(
        self,
        trial_name=None,
        osim_scaled_model_path=None,
        trc_path=None,
        remove_patella=True,
    ):

        # check if self.osim_scaled_model_path variable exists
        if osim_scaled_model_path is not None:
            self.osim_scaled_model_path = osim_scaled_model_path

        # logger.info(f"Using scaled opensim model: {self.osim_scaled_model_path}")

        # check if patella_removed model exists
        if remove_patella:
            self.osim_scaled_model_path = removePatella(self.osim_scaled_model_path)
            # logger.info(f"Patella removed from model.")

        if trial_name is not None:
            self.trial_name = trial_name

        if trc_path is not None:
            self.trc_path = trc_path

        self.ik_dir = os.path.join(self.osim_dir, "IK", self.trial_name)
        os.makedirs(self.ik_dir, exist_ok=True)

        self.ik_path = runIKTool(
            pathGenericSetupFile=self.opensim_ik_setup_path_generic,
            pathScaledModel=self.osim_scaled_model_path,
            pathTRCFile=self.trc_path,
            pathOutputFolder=self.ik_dir,
        )

        return self.ik_path


# Standalone functions


def smpl_to_smplx_verts(
    root_orient, root_trans, body_pose, betas, smpl_model_path=None
):
    """
    Parameters (copied from smplx.forward())
          ----------
          root_trans : B x T x 3
          root_orient : B x T x 3
          body_pose : B x T x J*3
          betas : B x D
    """

    B, T, _ = root_trans.shape

    if smpl_model_path is None:
        smpl_model_path = defaults()["SMPL_NEUTRAL_PATH"]

    smpl_to_smplx_map_path = defaults()["SMPL_TO_SMPLX_MAP_PATH"]

    with open(smpl_to_smplx_map_path, "rb") as file:
        smpl_to_smplx_dict = pickle.load(file)
    smpl_to_smplx_map = (
        torch.from_numpy(smpl_to_smplx_dict["matrix"]).float().to(root_orient.device)
    )

    # smpl_to_smplx_map_sparse = torch.sparse.FloatTensor(
    #     torch.LongTensor(np.vstack((smpl_to_smplx_map.row, smpl_to_smplx_map.col))),  # Indices
    #     torch.FloatTensor(smpl_to_smplx_map.data),  # Values
    #     torch.Size(smpl_to_smplx_map.shape)  # Size of the sparse matrix
    #     )

    smpl_model, _ = load_smpl_body_model(
        path=smpl_model_path,
        batch_size=B * T,
        num_betas=10,
        model_type="smpl",
        use_vtx_selector=True,
        device=root_orient.device,
        fit_gender="neutral",
        npz_hack=False,
    )

    output = utils_optim.pred_smpl(
        body_model=smpl_model,
        trans=root_trans,
        root_orient=root_orient,
        body_pose=body_pose,
        betas=betas,
    )

    vertices = output["points3d"]

    vertices_smplx = torch.torch.einsum("ijkl,km->ijml", vertices, smpl_to_smplx_map)

    openpose_markers = output["joints3d_op"]

    return vertices_smplx, openpose_markers


def smplx_to_markers(smplx_verts, openpose_markers):
    T = openpose_markers.shape[1]

    def get_vertices(vertex_idx, vertices):
        return np.moveaxis(
            np.array([vertices[:, vertex, :] for vertex in vertex_idx.values()]),
            (0, 1),
            (1, 0),
        )

    marker_names = list(AUGMENTED_VERTICES_INDEX_DICT.keys()) + OPENPOSE_VERTICES_NAME
    marker_positions = np.hstack(
        (
            get_vertices(
                AUGMENTED_VERTICES_INDEX_DICT,
                smplx_verts[0, :, :, :].detach().cpu().numpy().reshape((T, -1, 3)),
            ),
            openpose_markers[0].detach().cpu().numpy().reshape((T, -1, 3)),
        )
    ).squeeze()
    return marker_names, marker_positions


# # # # # # # # # # # # # # # # # # # #
# Modified from opencap-core repository
# # # # # # # # # # # # # # # # # # # #


# %% Scaling.
def runScaleTool(
    pathGenericSetupFile,
    pathGenericModel,
    subjectMass,
    pathTRCFile,
    timeRange,
    pathOutputFolder,
    scaledModelName="not_specified",
    subjectHeight=0,
    createModelWithContacts=False,
    fixed_markers=False,
    suffix_model="",
):

    dirGenericModel, scaledModelNameA = os.path.split(pathGenericModel)

    # Paths.
    if scaledModelName == "not_specified":
        scaledModelName = scaledModelNameA[:-5] + "_scaled"
    pathOutputModel = os.path.join(pathOutputFolder, scaledModelName + ".osim")
    pathOutputMotion = os.path.join(pathOutputFolder, scaledModelName + ".mot")
    pathOutputSetup = os.path.join(
        pathOutputFolder, "Setup_Scale_" + scaledModelName + ".xml"
    )
    pathUpdGenericModel = os.path.join(
        pathOutputFolder, scaledModelNameA[:-5] + "_generic.osim"
    )

    # Marker set.
    _, setupFileName = os.path.split(pathGenericSetupFile)
    if "Lai" in scaledModelName or "Rajagopal" in scaledModelName:
        if "Mocap" in setupFileName:
            markerSetFileName = "RajagopalModified2016_markers_mocap{}.xml".format(
                suffix_model
            )
        elif "openpose" in setupFileName:
            markerSetFileName = "RajagopalModified2016_markers_openpose.xml"
        elif "mmpose" in setupFileName:
            markerSetFileName = "RajagopalModified2016_markers_mmpose.xml"
        elif "SMPL" in setupFileName:
            markerSetFileName = "LaiUhlrich2022_markers_SMPL.xml"
        else:
            if fixed_markers:
                markerSetFileName = "RajagopalModified2016_markers_augmenter_fixed.xml"
            else:
                markerSetFileName = (
                    "RajagopalModified2016_markers_augmenter{}.xml".format(suffix_model)
                )
    elif "gait2392" in scaledModelName:
        if "Mocap" in setupFileName:
            markerSetFileName = "gait2392_markers_mocap.xml"
        else:
            markerSetFileName = "gait2392_markers_augmenter.xml"
    else:
        raise ValueError("Unknown model type: scaling")
    pathMarkerSet = os.path.join(dirGenericModel, markerSetFileName)

    # Add the marker set to the generic model and save that updated model.
    opensim.Logger.setLevelString("error")
    genericModel = opensim.Model(pathGenericModel)
    markerSet = opensim.MarkerSet(pathMarkerSet)
    genericModel.set_MarkerSet(markerSet)
    genericModel.printToXML(pathUpdGenericModel)

    # Time range.
    timeRange_os = opensim.ArrayDouble(timeRange[0], 0)
    timeRange_os.insert(1, timeRange[-1])

    # Setup scale tool.
    scaleTool = opensim.ScaleTool(pathGenericSetupFile)
    scaleTool.setName(scaledModelName)
    scaleTool.setSubjectMass(subjectMass)
    scaleTool.setSubjectHeight(subjectHeight)
    genericModelMaker = scaleTool.getGenericModelMaker()
    genericModelMaker.setModelFileName(pathUpdGenericModel)
    modelScaler = scaleTool.getModelScaler()
    modelScaler.setMarkerFileName(pathTRCFile)
    modelScaler.setOutputModelFileName("")
    modelScaler.setOutputScaleFileName("")
    modelScaler.setTimeRange(timeRange_os)
    markerPlacer = scaleTool.getMarkerPlacer()
    markerPlacer.setMarkerFileName(pathTRCFile)
    markerPlacer.setOutputModelFileName(pathOutputModel)
    markerPlacer.setOutputMotionFileName(pathOutputMotion)
    markerPlacer.setOutputMarkerFileName("")
    markerPlacer.setTimeRange(timeRange_os)

    # Disable tasks of dofs that are locked and markers that are not present.
    model = opensim.Model(pathUpdGenericModel)
    coordNames = []
    for coord in model.getCoordinateSet():
        if not coord.getDefaultLocked():
            coordNames.append(coord.getName())
    modelMarkerNames = [marker.getName() for marker in model.getMarkerSet()]

    for task in markerPlacer.getIKTaskSet():
        # Remove IK tasks for dofs that are locked or don't exist.
        if (
            task.getName() not in coordNames
            and task.getConcreteClassName() == "IKCoordinateTask"
        ):
            task.setApply(False)
            print("{} is a locked coordinate - ignoring IK task".format(task.getName()))
        # Remove Marker tracking tasks for markers not in model.
        if (
            task.getName() not in modelMarkerNames
            and task.getConcreteClassName() == "IKMarkerTask"
        ):
            task.setApply(False)
            print("{} is not in model - ignoring IK task".format(task.getName()))

    # Remove measurements from measurement set when markers don't exist.
    # Disable entire measurement if no complete marker pairs exist.
    measurementSet = modelScaler.getMeasurementSet()
    for meas in measurementSet:
        mkrPairSet = meas.getMarkerPairSet()
        iMkrPair = 0
        while iMkrPair < meas.getNumMarkerPairs():
            mkrPairNames = [mkrPairSet.get(iMkrPair).getMarkerName(i) for i in range(2)]
            if any([mkr not in modelMarkerNames for mkr in mkrPairNames]):
                mkrPairSet.remove(iMkrPair)
                print(
                    "{} or {} not in model. Removing associated \
                      MarkerPairSet from {}.".format(
                        mkrPairNames[0], mkrPairNames[1], meas.getName()
                    )
                )
            else:
                iMkrPair += 1
            if meas.getNumMarkerPairs() == 0:
                meas.setApply(False)
                print(
                    "There were no marker pairs in {}, so this measurement \
                      is not applied.".format(
                        meas.getName()
                    )
                )
    # Run scale tool.
    scaleTool.printToXML(pathOutputSetup)
    command = "opensim-cmd -o error" + " run-tool " + pathOutputSetup
    os.system(command)

    # Sanity check
    scaled_model = opensim.Model(pathOutputModel)
    bodySet = scaled_model.getBodySet()
    nBodies = bodySet.getSize()
    scale_factors = np.zeros((nBodies, 3))
    for i in range(nBodies):
        bodyName = bodySet.get(i).getName()
        body = bodySet.get(bodyName)
        attached_geometry = body.get_attached_geometry(0)
        scale_factors[i, :] = attached_geometry.get_scale_factors().to_numpy()
    diff_scale = np.max(np.max(scale_factors, axis=0) - np.min(scale_factors, axis=0))
    # A difference in scaling factor larger than 1 would indicate that a
    # segment (e.g., humerus) would be more than twice as large as its generic
    # counterpart, whereas another segment (e.g., pelvis) would have the same
    # size as the generic segment. This is very unlikely, but might occur when
    # the camera calibration went wrong (i.e., bad extrinsics).
    if diff_scale > 1:
        exception = "Musculoskeletal model scaling failed; the segment sizes are not anthropometrically realistic. It is very likely that the camera calibration went wrong. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration."
        raise Exception(exception, exception)

    return pathOutputModel


# %% Inverse kinematics.
def runIKTool(
    pathGenericSetupFile,
    pathScaledModel,
    pathTRCFile,
    pathOutputFolder,
    timeRange=None,
    IKFileName="not_specified",
):

    # Paths
    if IKFileName == "not_specified":
        _, IKFileName = os.path.split(pathTRCFile)
        IKFileName = IKFileName[:-4]
    pathOutputMotion = os.path.join(pathOutputFolder, IKFileName + ".mot")
    pathOutputSetup = os.path.join(pathOutputFolder, "Setup_IK_" + IKFileName + ".xml")

    # Setup IK tool.
    opensim.Logger.setLevelString("error")
    IKTool = opensim.InverseKinematicsTool(pathGenericSetupFile)
    IKTool.setName(IKFileName)
    IKTool.set_model_file(pathScaledModel)
    IKTool.set_marker_file(pathTRCFile)
    # Generic setup XML ships with a fixed time_range (e.g. ~13.45 s). If callers omit
    # timeRange, OpenSim would truncate IK to that window while TRC / video match full trial.
    if timeRange is None or len(timeRange) < 2:
        trc = utils_trc.TRCFile(pathTRCFile)
        if trc.num_frames < 1:
            raise ValueError(f"TRC has no frames, cannot infer IK time range: {pathTRCFile}")
        timeRange = [float(trc.time[0]), float(trc.time[-1])]
        logger.info(
            "IK time range from TRC: {:.6f} .. {:.6f} s ({} frames, data_rate={} Hz)",
            timeRange[0],
            timeRange[1],
            trc.num_frames,
            trc.data_rate,
        )
    else:
        timeRange = [float(timeRange[0]), float(timeRange[-1])]
    IKTool.set_time_range(0, timeRange[0])
    IKTool.set_time_range(1, timeRange[1])
    IKTool.setResultsDir(pathOutputFolder)
    IKTool.set_report_errors(True)
    IKTool.set_report_marker_locations(False)
    IKTool.set_output_motion_file(pathOutputMotion)
    IKTool.printToXML(pathOutputSetup)
    command = "opensim-cmd -o error" + " run-tool " + pathOutputSetup
    os.system(command)

    return pathOutputMotion


# %% Remove patella to speed up IK.
def removePatella(path_model):
    # To make IK faster, we remove the patellas and their constraints from the
    # model. Constraints make the IK problem more difficult, and the patellas
    # are not used in the IK solution for this particular model. Since muscles
    # are attached to the patellas, we also remove all muscles.

    opensim.Logger.setLevelString("error")
    model = opensim.Model(path_model)
    # Remove all actuators.
    forceSet = model.getForceSet()
    forceSet.setSize(0)
    # Remove patellofemoral constraints.
    constraintSet = model.getConstraintSet()
    patellofemoral_constraints = [
        "patellofemoral_knee_angle_r_con",
        "patellofemoral_knee_angle_l_con",
    ]
    for patellofemoral_constraint in patellofemoral_constraints:
        i = constraintSet.getIndex(patellofemoral_constraint, 0)
        constraintSet.remove(i)
    # Remove patella bodies.
    bodySet = model.getBodySet()
    patella_bodies = ["patella_r", "patella_l"]
    for patella in patella_bodies:
        i = bodySet.getIndex(patella, 0)
        bodySet.remove(i)
    # Remove patellofemoral joints.
    jointSet = model.getJointSet()
    patellofemoral_joints = ["patellofemoral_r", "patellofemoral_l"]
    for patellofemoral in patellofemoral_joints:
        i = jointSet.getIndex(patellofemoral, 0)
        jointSet.remove(i)
    # Print the model to a new file.
    model.finalizeConnections
    model.initSystem()
    pathScaledModelWithoutPatella = path_model.replace(".osim", "_no_patella.osim")
    model.printToXML(pathScaledModelWithoutPatella)

    return pathScaledModelWithoutPatella


# %%
def runIDTool(
    pathGenericSetupFileID,
    pathGenericSetupFileEL,
    pathGRFFile,
    pathScaledModel,
    pathIKFile,
    timeRange,
    pathOutputFolder,
    filteringFrequency=10,
    IKFileName="not_specified",
):

    # Paths
    if IKFileName == "not_specified":
        _, IKFileName = os.path.split(pathIKFile)
        IKFileName = IKFileName[:-4]

    pathOutputSetupEL = os.path.join(
        pathOutputFolder, "Setup_EL_" + IKFileName + ".xml"
    )
    pathOutputSetupID = os.path.join(
        pathOutputFolder, "Setup_ID_" + IKFileName + ".xml"
    )

    # External loads
    opensim.Logger.setLevelString("error")
    ELTool = opensim.ExternalLoads(pathGenericSetupFileEL, True)
    ELTool.setDataFileName(pathGRFFile)
    ELTool.setName(IKFileName)
    ELTool.printToXML(pathOutputSetupEL)

    # ID
    IDTool = opensim.InverseDynamicsTool(pathGenericSetupFileID)
    IDTool.setModelFileName(pathScaledModel)
    IDTool.setName(IKFileName)
    IDTool.setStartTime(timeRange[0])
    IDTool.setEndTime(timeRange[-1])
    IDTool.setExternalLoadsFileName(pathOutputSetupEL)
    IDTool.setCoordinatesFileName(pathIKFile)
    IDTool.setLowpassCutoffFrequency(filteringFrequency)
    IDTool.setResultsDir(pathOutputFolder)
    IDTool.setOutputGenForceFileName(IKFileName + ".sto")
    IDTool.printToXML(pathOutputSetupID)
    command = "opensim-cmd -o error" + " run-tool " + pathOutputSetupID
    os.system(command)


# %% Might be outdated.
def addOpenPoseMarkersTool(
    pathModel,
    adjustLocationHipAnkle=True,
    hipOffsetX=0.036,
    hipOffsetZ=0.0175,
    ankleOffsetX=0.007,
):
    """
    This script adds virtual markers corresponding to the OpenPose markers
    to the OpenSim model. For most markers, we assume that the OpenPose
    markers correspond to the joint centers. For some markers, the
    locations have been hand-tuned for one subject.
    """

    pathModelFolder, modelName = os.path.split(pathModel)

    # Methods specific to each marker.
    markersInfo = {
        "RHip": {"method": "jointCenter", "joint": "hip_r", "parent": "pelvis"},
        "LHip": {"method": "jointCenter", "joint": "hip_l", "parent": "pelvis"},
        "midHip": {
            "method": "mid",
            "references": "jointCenters",
            "reference_jointCenter_A": "hip_r",
            "reference_jointCenter_B": "hip_l",
        },
        "RElbow": {"method": "jointCenter", "joint": "elbow_r", "parent": "humerus_r"},
        "LElbow": {"method": "jointCenter", "joint": "elbow_l", "parent": "humerus_l"},
        "RWrist": {
            "method": "jointCenter",
            "joint": "radius_hand_r",
            "parent": "hand_r",
        },
        "LWrist": {
            "method": "jointCenter",
            "joint": "radius_hand_l",
            "parent": "hand_l",
        },
        "RShoulder": {
            "method": "jointCenter",
            "joint": "acromial_r",
            "parent": "torso",
        },
        "LShoulder": {
            "method": "jointCenter",
            "joint": "acromial_l",
            "parent": "torso",
        },
        "RKnee": {
            "method": "jointCenter",
            "joint": "walker_knee_r",
            "parent": "tibia_r",
        },
        "LKnee": {
            "method": "jointCenter",
            "joint": "walker_knee_l",
            "parent": "tibia_l",
        },
        "Neck": {
            "method": "mid",
            "references": "jointCenters",
            "reference_jointCenter_A": "acromial_r",
            "reference_jointCenter_B": "acromial_l",
        },
    }
    # Values manually set.
    markersInfo["RHeel"] = {
        "method": "location",
        "parent": "calcn_r",
        "location": np.array([0.018205985755086355, 0.01, -0.020246741086146904]),
    }
    markersInfo["LHeel"] = {
        "method": "location",
        "parent": "calcn_l",
        "location": np.array([0.018205985755086355, 0.01, 0.020246741086146904]),
    }
    markersInfo["RSmallToe"] = {
        "method": "location",
        "parent": "toes_r",
        "location": np.array([0.0214658, 0.002, 0.0394135]),
    }
    markersInfo["LSmallToe"] = {
        "method": "location",
        "parent": "toes_l",
        "location": np.array([0.0214658, 0.002, -0.0394135]),
    }
    markersInfo["RBigToe"] = {
        "method": "location",
        "parent": "toes_r",
        "location": np.array([0.0487748, 0.002, -0.0162651]),
    }
    markersInfo["LBigToe"] = {
        "method": "location",
        "parent": "toes_l",
        "location": np.array([0.0487748, 0.002, 0.0162651]),
    }
    markersInfo["RAnkle"] = {
        "method": "location",
        "parent": "tibia_r",
        "location": np.array([-0.007, -0.38, 0.0075]),
    }
    markersInfo["LAnkle"] = {
        "method": "location",
        "parent": "tibia_l",
        "location": np.array([-0.007, -0.38, -0.0075]),
    }

    # Add OpenPose markers and print new model
    opensim.Logger.setLevelString("error")
    model = opensim.Model(pathModel)
    bodySet = model.get_BodySet()
    jointSet = model.get_JointSet()
    markerSet = model.get_MarkerSet()
    for marker in markersInfo:

        if markersInfo[marker]["method"] == "marker":
            referenceMarker = markerSet.get(markersInfo[marker]["reference_marker"])
            parentFrame = referenceMarker.getParentFrameName()
            location = referenceMarker.get_location()

        if markersInfo[marker]["method"] == "jointCenter":
            joint = jointSet.get(markersInfo[marker]["joint"])
            if (
                (marker == "RKnee" and markersInfo[marker]["parent"] == "tibia_r")
                or (marker == "LKnee" and markersInfo[marker]["parent"] == "tibia_l")
                or (
                    marker == "RKnee"
                    and markersInfo[marker]["parent"] == "sagittal_articulation_frame_r"
                )
                or (
                    marker == "LKnee"
                    and markersInfo[marker]["parent"] == "sagittal_articulation_frame_l"
                )
                or (marker == "RWrist" and markersInfo[marker]["parent"] == "hand_r")
                or (marker == "LWrist" and markersInfo[marker]["parent"] == "hand_l")
            ):
                frame = joint.get_frames(1)
            else:
                frame = joint.get_frames(0)
            assert frame.getName()[:-7] == markersInfo[marker]["parent"]
            location = frame.get_translation()

            if adjustLocationHipAnkle:
                if marker == "RHip" or marker == "LHip":

                    # Get scale factor based on saved factors from scaling.
                    body = bodySet.get(markersInfo[marker]["parent"])
                    attached_geometry = body.get_attached_geometry(0)
                    scale_factors = attached_geometry.get_scale_factors().to_numpy()
                    location_np = location.to_numpy()

                    # After comparing triangulated OpenPose markers and
                    # mocap-based markers, it appears that the hip OpenPose
                    # markers should be located more forward and lateral.
                    location_adj_np = np.copy(location_np)
                    location_adj_np[0] += hipOffsetX * scale_factors[0]
                    if marker == "RHip":
                        location_adj_np[2] += hipOffsetZ * scale_factors[2]
                    elif marker == "LHip":
                        location_adj_np[2] -= hipOffsetZ * scale_factors[2]

                    location = opensim.Vec3(location_adj_np)

            parentFrame = "/bodyset/" + markersInfo[marker]["parent"]

        elif markersInfo[marker]["method"] == "mid":
            if markersInfo[marker]["references"] == "markers":
                referenceMarker_A = markerSet.get(
                    markersInfo[marker]["reference_marker_A"]
                )
                parentFrame_A = referenceMarker_A.getParentFrameName()
                location_A = referenceMarker_A.get_location().to_numpy()
                referenceMarker_B = markerSet.get(
                    markersInfo[marker]["reference_marker_B"]
                )
                parentFrame_B = referenceMarker_B.getParentFrameName()
                location_B = referenceMarker_B.get_location().to_numpy()
                parentFrame = parentFrame_A

            elif markersInfo[marker]["references"] == "jointCenters":
                referenceJoint_A = jointSet.get(
                    markersInfo[marker]["reference_jointCenter_A"]
                )
                frame_A = referenceJoint_A.get_frames(0)
                parentFrame_A = frame_A.getName()
                location_A = frame_A.get_translation().to_numpy()
                referenceJoint_B = jointSet.get(
                    markersInfo[marker]["reference_jointCenter_B"]
                )
                frame_B = referenceJoint_B.get_frames(0)
                parentFrame_B = frame_B.getName()
                location_B = frame_B.get_translation().to_numpy()
                parentFrame = "/bodyset/" + frame_A.getName()[:-7]
            assert parentFrame_A == parentFrame_B, "error parent frames"
            location = opensim.Vec3((location_A + location_B) / 2)

        elif markersInfo[marker]["method"] == "location":
            location_np = markersInfo[marker]["location"]
            # Get scale factor based on saved scale factors from scaling.
            body = bodySet.get(markersInfo[marker]["parent"])
            attached_geometry = body.get_attached_geometry(0)
            scale_factors = attached_geometry.get_scale_factors().to_numpy()
            location_np_scaled = location_np * scale_factors
            location = opensim.Vec3(location_np_scaled)
            parentFrame = "/bodyset/" + markersInfo[marker]["parent"]

        if "y_pos" in markersInfo[marker]:
            location_t = location.to_numpy()
            referenceMarker_ypos = markerSet.get(markersInfo[marker]["y_pos"])
            location_ref = referenceMarker_ypos.get_location().to_numpy()
            location_t[1] = location_ref[1]
            location = opensim.Vec3(location_t)

        newMkr = opensim.Marker()
        newMkr.setName(marker)
        newMkr.setParentFrameName(parentFrame)
        newMkr.set_location(location)
        model.addMarker(newMkr)

    model.finalizeConnections
    model.initSystem()
    pathNewModel = os.path.join(pathModelFolder, modelName[:-5] + "_OpenPose.osim")
    model.printToXML(pathNewModel)


# %% This takes model and IK and generates a json of body transforms that can
# be passed to the webapp visualizer
# def generateVisualizerJson(
#     modelPath, ikPath, jsonOutputPath, statesInDegrees=True, vertical_offset=None
# ):

#     opensim.Logger.setLevelString("error")
#     model = opensim.Model(modelPath)
#     bodyset = model.getBodySet()

#     coords = model.getCoordinateSet()
#     nCoords = coords.getSize()
#     coordNames = [coords.get(i).getName() for i in range(nCoords)]

#     # load IK
#     stateTable = opensim.TimeSeriesTable(ikPath)
#     stateNames = stateTable.getColumnLabels()
#     stateTime = stateTable.getIndependentColumn()
#     try:
#         inDegrees = stateTable.getTableMetaDataAsString("inDegrees") == "yes"
#     except:
#         inDegrees = statesInDegrees
#         print(
#             "using statesInDegrees variable, which says statesInDegrees is "
#             + str(statesInDegrees)
#         )
#     q = np.zeros((len(stateTime), nCoords))

#     stateNamesOut = []
#     for col in stateNames:
#         if "activation" in col:
#             stateTable.removeColumn(col)
#         elif col[0] == "/" and any(
#             ["jointset" not in col, "value" not in col]
#         ):  # full state path
#             stateTable.removeColumn(col)
#         else:
#             coordCol = [i for i, c in enumerate(coordNames) if c in col][0]
#             coordName = col
#             if col[0] == "/":  # if full state path
#                 temp = col[: col.rfind("/")]
#                 coordName = temp[temp.rfind("/") + 1 :]
#             for t in range(len(stateTime)):
#                 qTemp = np.asarray(stateTable.getDependentColumn(col)[t])
#                 if coords.get(coordName).getMotionType() == 1 and inDegrees:  # rotation
#                     qTemp = np.deg2rad(qTemp)
#                 if "pelvis_ty" in col and not (vertical_offset is None):
#                     qTemp += vertical_offset  # Add the vertical offset to move the model up
#                 q[t, coordCol] = qTemp
#             stateNamesOut.append(
#                 coordName
#             )  # This is always just coord - never full path

#     # We may have deleted some columns
#     stateNames = stateNamesOut

#     state = model.initSystem()

#     # Create state Y map
#     yNames = opensim.createStateVariableNamesInSystemOrder(model)
#     systemStateInds = []
#     for stateName in stateNames:
#         stateIdx = np.squeeze(np.argwhere([stateName + "/value" in y for y in yNames]))
#         systemStateInds.append(stateIdx)

#     # Loop over time and bodies
#     visualizeDict = {}
#     visualizeDict["time"] = stateTime
#     visualizeDict["bodies"] = {}

#     for body in bodyset:
#         visualizeDict["bodies"][body.getName()] = {}
#         attachedGeometries = []

#         # Ayman said that meshes could get attached to model in different ways than
#         # this, so this isn't most general sol'n, but should work for now
#         thisFrame = opensim.Frame.safeDownCast(body)
#         nGeometries = thisFrame.getPropertyByName("attached_geometry").size()

#         for iGeom in range(nGeometries):
#             attached_geometry = body.get_attached_geometry(iGeom)
#             if attached_geometry.getConcreteClassName() == "Mesh":
#                 thisMesh = opensim.Mesh.safeDownCast(attached_geometry)
#                 attachedGeometries.append(thisMesh.getGeometryFilename())
#         visualizeDict["bodies"][body.getName()][
#             "attachedGeometries"
#         ] = attachedGeometries

#         scale_factors = attached_geometry.get_scale_factors().to_numpy()
#         visualizeDict["bodies"][body.getName()]["scaleFactors"] = scale_factors.tolist()

#         # init body translation and rotations dictionaries

#         visualizeDict["bodies"][body.getName()]["rotation"] = []
#         visualizeDict["bodies"][body.getName()]["translation"] = []

#     for iTime, time in enumerate(stateTime):
#         yVec = np.zeros((state.getNY())).tolist()
#         for i in range(nCoords):
#             yVec[systemStateInds[i]] = q[iTime, i]
#         state.setY(opensim.Vector(yVec))

#         model.realizePosition(state)

#         # get body translations and rotations in ground
#         for body in bodyset:
#             # This gives us body transform to opensim body frame, which isn't nec.
#             # geometry origin. Ayman said getting transform to Geometry::Mesh is safest
#             # but we don't have access to it thru API and Ayman said what we're doing
#             # is OK for now
#             visualizeDict["bodies"][body.getName()]["rotation"].append(
#                 body.getTransformInGround(state)
#                 .R()
#                 .convertRotationToBodyFixedXYZ()
#                 .to_numpy()
#                 .tolist()
#             )
#             visualizeDict["bodies"][body.getName()]["translation"].append(
#                 body.getTransformInGround(state).T().to_numpy().tolist()
#             )

#     with open(jsonOutputPath, "w") as f:
#         json.dump(visualizeDict, f)

#     return


# %% Load storage and output as dataframe or numpy
def load_storage(file_path, outputFormat="numpy"):
    table = opensim.TimeSeriesTable(file_path)
    data = table.getMatrix().to_numpy()
    time = np.asarray(table.getIndependentColumn()).reshape(-1, 1)
    data = np.hstack((time, data))
    headers = ["time"] + list(table.getColumnLabels())

    if outputFormat == "numpy":
        return data, headers
    elif outputFormat == "dataframe":
        return pd.DataFrame(data, columns=headers)
    else:
        return None
