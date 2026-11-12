import os
import opensim
import numpy as np
import glob
import json


# %% This takes model and IK and generates a json of body transforms that can
# be passed to the webapp visualizer
def generateVisualizerJson(
    modelPath, ikPath, jsonOutputPath, statesInDegrees=True, vertical_offset=None
):

    opensim.Logger.setLevelString("error")
    model = opensim.Model(modelPath)
    bodyset = model.getBodySet()

    coords = model.getCoordinateSet()
    nCoords = coords.getSize()
    coordNames = [coords.get(i).getName() for i in range(nCoords)]

    # load IK
    stateTable = opensim.TimeSeriesTable(ikPath)
    stateNames = stateTable.getColumnLabels()
    stateTime = list(stateTable.getIndependentColumn())
    try:
        inDegrees = stateTable.getTableMetaDataAsString("inDegrees") == "yes"
    except:
        inDegrees = statesInDegrees
        print(
            "using statesInDegrees variable, which says statesInDegrees is "
            + str(statesInDegrees)
        )
    q = np.zeros((len(stateTime), nCoords))

    stateNamesOut = []
    for col in stateNames:
        if "activation" in col:
            stateTable.removeColumn(col)
        elif col[0] == "/" and any(
            ["jointset" not in col, "value" not in col]
        ):  # full state path
            stateTable.removeColumn(col)
        else:
            coordCol = [i for i, c in enumerate(coordNames) if c in col][0]
            coordName = col
            if col[0] == "/":  # if full state path
                temp = col[: col.rfind("/")]
                coordName = temp[temp.rfind("/") + 1 :]
            for t in range(len(stateTime)):
                qTemp = np.asarray(stateTable.getDependentColumn(col)[t])
                if coords.get(coordName).getMotionType() == 1 and inDegrees:  # rotation
                    qTemp = np.deg2rad(qTemp)
                if "pelvis_ty" in col and not (vertical_offset is None):
                    qTemp += (
                        vertical_offset  # Add the vertical offset to move the model up
                    )
                q[t, coordCol] = qTemp
            stateNamesOut.append(
                coordName
            )  # This is always just coord - never full path

    # We may have deleted some columns
    stateNames = stateNamesOut

    state = model.initSystem()

    # Create state Y map
    yNames = opensim.createStateVariableNamesInSystemOrder(model)
    systemStateInds = []
    for stateName in stateNames:
        stateIdx = np.squeeze(np.argwhere([stateName + "/value" in y for y in yNames]))
        systemStateInds.append(stateIdx)

    # Loop over time and bodies
    visualizeDict = {}
    visualizeDict["time"] = stateTime
    visualizeDict["bodies"] = {}

    for body in bodyset:
        visualizeDict["bodies"][body.getName()] = {}
        attachedGeometries = []

        # Ayman said that meshes could get attached to model in different ways than
        # this, so this isn't most general sol'n, but should work for now
        thisFrame = opensim.Frame.safeDownCast(body)
        nGeometries = thisFrame.getPropertyByName("attached_geometry").size()

        for iGeom in range(nGeometries):
            attached_geometry = body.get_attached_geometry(iGeom)
            if attached_geometry.getConcreteClassName() == "Mesh":
                thisMesh = opensim.Mesh.safeDownCast(attached_geometry)
                attachedGeometries.append(thisMesh.getGeometryFilename())
        visualizeDict["bodies"][body.getName()][
            "attachedGeometries"
        ] = attachedGeometries

        scale_factors = attached_geometry.get_scale_factors().to_numpy()
        visualizeDict["bodies"][body.getName()]["scaleFactors"] = scale_factors.tolist()

        # init body translation and rotations dictionaries

        visualizeDict["bodies"][body.getName()]["rotation"] = []
        visualizeDict["bodies"][body.getName()]["translation"] = []

    for iTime, time in enumerate(stateTime):
        yVec = np.zeros((state.getNY())).tolist()
        for i in range(nCoords):
            yVec[systemStateInds[i]] = q[iTime, i]
        state.setY(opensim.Vector(yVec))

        model.realizePosition(state)

        # get body translations and rotations in ground
        for body in bodyset:
            # This gives us body transform to opensim body frame, which isn't nec.
            # geometry origin. Ayman said getting transform to Geometry::Mesh is safest
            # but we don't have access to it thru API and Ayman said what we're doing
            # is OK for now
            visualizeDict["bodies"][body.getName()]["rotation"].append(
                body.getTransformInGround(state)
                .R()
                .convertRotationToBodyFixedXYZ()
                .to_numpy()
                .tolist()
            )
            visualizeDict["bodies"][body.getName()]["translation"].append(
                body.getTransformInGround(state).T().to_numpy().tolist()
            )

    with open(jsonOutputPath, "w") as f:
        json.dump(visualizeDict, f)

    return jsonOutputPath
