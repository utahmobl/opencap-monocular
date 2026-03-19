#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 00:03:12 2024

@author: opencap
"""

import pkg_resources
import os


repo_osim_dir = pkg_resources.resource_filename("utils", "opensim")
repo_dir = os.path.abspath(os.path.join(repo_osim_dir, "../", "../"))


def defaults():
    defaults = {
        "OPENSIM_MODEL_PATH": os.path.join(
            repo_osim_dir, "Model", "LaiUhlrich2022.osim"
        ),
        "OPENSIM_SCALING_SETUP_PATH": os.path.join(
            repo_osim_dir, "Scaling", "Setup_scaling_LaiUhlrich2022_SMPL.xml"
        ),
        "OPENSIM_IK_SETUP_PATH": os.path.join(repo_osim_dir, "IK", "Setup_IK_SMPL.xml"),
        "SMPL_NEUTRAL_PATH": os.path.join(
            repo_dir, "WHAM", "dataset", "body_models", "smpl", "SMPL_NEUTRAL.pkl"
        ),
        "SMPL_TO_SMPLX_MAP_PATH": os.path.join(
            repo_dir, "WHAM", "dataset", "model_transfer", "smplx_to_smpl.pkl"
        ),
    }

    return defaults
