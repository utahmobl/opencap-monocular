#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:16:57 2024

@author: opencap
"""


import joblib
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import cv2

import visualize_2d

basePath = osp.dirname(osp.abspath(__file__))
output_pth = osp.join(basePath, "../", "output", "walking4", "case0", "walking4")


dataset = "TopDownCocoWholeBodyDataset"
tracking_results = joblib.load(osp.join(output_pth, "tracking_results_wholebody.pth"))

# dataset = "TopDownCocoDataset"
# tracking_results = joblib.load(osp.join(output_pth, 'tracking_results_coco17.pth'))

# %%

frame = np.round(4.1 * 60)
slice_idx = np.argwhere(tracking_results[1]["frame_id"] == frame)[0][0]

keypoint_subset = tracking_results[1]["keypoints"][slice_idx, :, :]

pose_results = []
pose_results.append(keypoint_subset)

img = visualize_2d.vis_keypoints(
    pose_results, [720, 1280], radius=6, thickness=3, kpt_score_thr=0.3, dataset=dataset
)

plt.imshow(img)

test = 1
