#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:20:04 2024

@author: opencap
"""

import sys
import os

import numpy as np
import torch
import kornia.geometry.conversions as conversions

import slahmr.slahmr.geometry.camera as camera
import utils.utils_optim as ut

import third_party_modified.ipman.stability.ground_losses as ipman_ground_losses
from utils.data.constants import OPENPOSE_VERTICES_NAME
from loguru import logger


keypoints_to_ignore = []
keypoints_to_use = [i for i in range(25) if i not in keypoints_to_ignore]

# torch.autograd.set_detect_anomaly(True)


class OptimizeExtrinsics:
    """
    Optimize the camera extrinsics (position and orientation) to better align
    3D SMPL keypoints with 2D image keypoints.

    This class uses differentiable optimization to find the best camera extrinsics
    that minimize reprojection error and satisfy various constraints.

    Attributes:
        design_vars: List of optimization variables
        n_frames: Number of frames in the optimization window
        iterations: Maximum number of optimization iterations
        conv_tol: Convergence tolerance for optimization
        static_cam: Whether to assume a static camera
        key2d_image: 2D keypoints from image detection
        key3d: 3D keypoints from SMPL model
        cam_center, cam_f: Camera intrinsics (principal point and focal length)
        cam_distortion_k, cam_distortion_p: Camera distortion parameters
        weights: Weights for different loss terms
        t, quat: Camera translation and rotation (as quaternion)
        smpl_model, beta, height: SMPL model and parameters (if optimizing height)
    """

    def __init__(
        self,
        R_world_to_cam_init,
        t_world_to_cam_init,
        key2d_image,
        key3d_smpl,
        intrinsics,
        height=None,
        smpl_model=None,
        beta=None,
        frame_range=None,
        weights=None,
        static_cam=True,
        iterations=20,
        printer=False,
    ):
        """
        Initialize the camera extrinsics optimizer.

        Args:
            R_world_to_cam_init: Initial camera rotation matrix (world to camera)
            t_world_to_cam_init: Initial camera translation vector (world to camera)
            key2d_image: 2D keypoints detected in image
            key3d_smpl: 3D keypoints from SMPL model
            intrinsics: Camera intrinsic parameters (fx, fy, cx, cy, distortion)
            height (optional): Subject height for constraint
            smpl_model (optional): SMPL model for height optimization
            beta (optional): SMPL body shape parameters
            frame_range (optional): Range of frames to optimize over
            weights (optional): Weights for different loss terms
            static_cam (bool): Whether to assume a static camera
            iterations (int): Maximum number of optimization iterations
            printer (bool): Whether to print debug information
        """

        # preallocate design vars list
        self.design_vars = []
        self.device = R_world_to_cam_init.device

        if frame_range is None:
            frame_range = range(t_world_to_cam_init.shape[0])
        self.n_frames = len(frame_range)

        self.iterations = iterations
        self.conv_tol = 0.001  # if doesn't change by this much in patience iterations
        self.early_stop_patience = 5
        self.early_stop_rel_improvement_pct = 1.0
        self.static_cam = static_cam

        self.key2d_image = key2d_image[frame_range, :, :].unsqueeze(0)
        self.key3d = key3d_smpl[frame_range, :, :].unsqueeze(
            0
        )  # only one body for now (T,N,3) -> (B, T, N, 3)

        self.cam_center = torch.tensor(
            [intrinsics["cx"], intrinsics["cy"]], dtype=torch.float32, device=self.device
        ).reshape(1, 2)

        self.cam_f = torch.tensor(
            [intrinsics["fx"], intrinsics["fy"]], dtype=torch.float32, device=self.device
        ).reshape(1, 2)

        # distortion
        if "k1" in intrinsics:
            self.cam_distortion_k = (
                torch.tensor(
                    [intrinsics["k1"], intrinsics["k2"], intrinsics["k3"]],
                    dtype=torch.float32,
                    device=self.device,
                )
                .reshape(1, 3)
                .repeat(self.n_frames, 1)
            )
        else:
            self.cam_distortion_k = None
        if "p1" in intrinsics:
            self.cam_distortion_p = (
                torch.tensor(
                    [intrinsics["p1"], intrinsics["p2"]],
                    dtype=torch.float32,
                    device=self.device,
                )
                .reshape(1, 2)
                .repeat(self.n_frames, 1)
            )
        else:
            self.cam_distortion_p = None

        if weights is None:  # Default weights
            self.weights = {
                "reprojection": 1,
                "quat_1": 1,
                "beta_change_regularization": 0.1,
                "height": 1,
            }
        else:
            self.weights = weights

        # initial guesses assuming fixed camera
        self.t = torch.mean(t_world_to_cam_init[frame_range, :], axis=0).squeeze()
        self.t.requires_grad = True
        self.design_vars.append(self.t)
        self.printer = printer
        self.quat = torch.mean(
            conversions.rotation_matrix_to_quaternion(
                R_world_to_cam_init[frame_range, :, :]
            ),
            axis=0,
        )
        self.quat.requires_grad = True
        self.design_vars.append(self.quat)

        # Precomputed values (must be at end of init)
        if "reprojection" in self.weights:
            self.confidence_selected_keypoints = (
                self.key2d_image[:, :, keypoints_to_use, 2]
                .unsqueeze(-1)
                .repeat(1, 1, 1, 2)
            )
            self.key2d_image_selected_keypoints = self.key2d_image[
                :, :, keypoints_to_use, :2
            ]
            self.reprojection_loss_init = (
                self.loss_reprojection().clone().detach().cpu() / 1000
            )

        if "height" in self.weights and height is not None:
            if smpl_model is None or beta is None:
                raise Exception(
                    "You are optimizing height w/o a SMPL model or betas provided."
                )
            self.smpl_model = smpl_model
            self.height = height
            self.beta_init = beta.clone()
            self.trans_0 = torch.zeros(1, 1, 3, dtype=torch.float32, device=self.device)
            self.root_orient_0 = torch.zeros(1, 1, 3, dtype=torch.float32, device=self.device)
            self.body_pose_0 = torch.zeros(1, 1, 69, dtype=torch.float32, device=self.device)

            # beta becomes a design var
            self.beta = beta
            self.beta.requires_grad = True
            self.design_vars.append(self.beta)

            # init value- comes last
            self.height_loss_init = self.loss_height().clone().detach().cpu() / 1000

    def loss_reprojection(self, scale=1):

        self.R = (
            conversions.quaternion_to_rotation_matrix(self.quat)
            .unsqueeze(0)
            .repeat(self.n_frames, 1, 1)
            .unsqueeze(0)
        )

        key2d_smpl = camera.reproject(
            self.key3d,
            self.R,
            self.t.reshape(1, 1, 3),
            self.cam_f,
            self.cam_center,
            cam_distortion_k=self.cam_distortion_k,
            cam_distortion_p=self.cam_distortion_p,
        )

        loss = (
            (
                self.confidence_selected_keypoints  # confidence weight
                * (
                    key2d_smpl[:, :, keypoints_to_use, :]
                    - self.key2d_image_selected_keypoints
                )
            )
            ** 2
        ).sum()

        if self.printer:
            print("reprojection loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_quat_1(self):
        loss = (torch.norm(self.quat) - 1) ** 2

        return loss

    def loss_height(self, scale=1):
        smpl_height = self.compute_smpl_height()
        if self.printer:
            print("smpl_height=", smpl_height.detach().cpu().numpy())

        loss = (self.height - smpl_height) ** 2

        return loss / scale

    def compute_smpl_height(self):
        smpl_0 = ut.pred_smpl(
            self.smpl_model,
            self.trans_0,
            self.root_orient_0,
            self.body_pose_0,
            self.beta,
        )
        heel_ind = 6664  # min y vert_id in neutral-posed smpl
        head_ind = 411  # max y vert_id in neutral-posed smpl
        height = torch.abs(
            smpl_0["verts3d_all"][0, 0, heel_ind, 1]
            - smpl_0["verts3d_all"][0, 0, head_ind, 1]
        )

        return height

    def loss_beta_change_regularization(self):
        loss = ((self.beta_init - self.beta) ** 2).sum()

        return loss

    def objective_function(self):
        loss = 0
        if "reprojection" in self.weights:
            loss += self.weights["reprojection"] * self.loss_reprojection(
                scale=self.reprojection_loss_init
            )
        if "quat_1" in self.weights:
            loss += self.weights["quat_1"] * self.loss_quat_1()
        if "height" in self.weights:
            loss += self.weights["height"] * self.loss_height(
                scale=self.height_loss_init
            )
        if "beta_change_regularization" in self.weights:
            loss += (
                self.weights["beta_change_regularization"]
                * self.loss_beta_change_regularization()
            )

        return loss

    def optimize(self):
        # Save initial state for fallback recovery
        t_init = self.t.clone().detach()
        quat_init = self.quat.clone().detach()
        initial_state = {"t": t_init, "quat": quat_init}
        if hasattr(self, "beta"):
            initial_state["beta"] = self.beta.clone().detach()

        last_good_state = {k: v.clone() for k, v in initial_state.items()}

        def save_good_state():
            last_good_state["t"] = self.t.clone().detach()
            last_good_state["quat"] = self.quat.clone().detach()
            if hasattr(self, "beta"):
                last_good_state["beta"] = self.beta.clone().detach()

        def restore_state(state):
            with torch.no_grad():
                self.t.copy_(state["t"])
                self.quat.copy_(state["quat"])
                if hasattr(self, "beta") and "beta" in state:
                    self.beta.copy_(state["beta"])

        nan_recovery_count = [0]
        max_nan_recoveries = 5

        # Initialize loss so self.last_loss is always safe to assign
        self.loss = torch.tensor(1e6, device=self.device, requires_grad=True)

        # Create an L-BFGS optimizer
        optimizer = torch.optim.LBFGS(
            self.design_vars,
            tolerance_change=self.conv_tol,
            line_search_fn="strong_wolfe",
        )

        # Define the closure function that reevaluates the model
        def closure():
            # Check for NaN/Inf in design variables before the forward pass
            vars_to_check = [("t", self.t), ("quat", self.quat)]
            if hasattr(self, "beta"):
                vars_to_check.append(("beta", self.beta))

            has_nan = any(not torch.isfinite(v).all() for _, v in vars_to_check)

            if has_nan:
                nan_recovery_count[0] += 1
                if nan_recovery_count[0] <= max_nan_recoveries:
                    logger.warning(
                        f"OptimizeExtrinsics: NaN/Inf in design vars at recovery "
                        f"#{nan_recovery_count[0]}, restoring last good state"
                    )
                    restore_state(last_good_state)
                else:
                    logger.warning(
                        f"OptimizeExtrinsics: too many NaN recoveries "
                        f"({nan_recovery_count[0]}), restoring initial state"
                    )
                    restore_state(initial_state)
                self.loss = torch.tensor(1e6, device=self.device, requires_grad=True)
                return self.loss

            optimizer.zero_grad()
            self.loss = self.objective_function()

            if not torch.isfinite(self.loss):
                nan_recovery_count[0] += 1
                logger.warning(
                    f"OptimizeExtrinsics: NaN/Inf loss at recovery "
                    f"#{nan_recovery_count[0]}, restoring last good state"
                )
                restore_state(last_good_state)
                self.loss = torch.tensor(1e6, device=self.device, requires_grad=True)
                return self.loss

            save_good_state()
            self.loss.backward(retain_graph=True)
            if self.printer:
                print("loss: ", self.loss.detach().cpu().numpy())
                for var in self.design_vars:
                    print("gradients: ", var.grad.detach().cpu().numpy())
            return self.loss

        # Optimization loop
        best_loss = None
        no_improvement_iters = 0
        for i in range(self.iterations):
            if nan_recovery_count[0] >= max_nan_recoveries:
                logger.warning(
                    f"OptimizeExtrinsics: stopping L-BFGS early at iteration {i} "
                    f"due to {nan_recovery_count[0]} NaN recoveries"
                )
                break
            optimizer.step(closure)
            self.last_loss = self.loss.clone()

            current_loss = float(self.last_loss.detach().cpu())
            relative_improvement_pct = 0.0
            if best_loss is not None:
                relative_improvement_pct = (
                    (best_loss - current_loss) / max(abs(best_loss), 1e-12)
                ) * 100.0

            if (
                best_loss is None
                or relative_improvement_pct > self.early_stop_rel_improvement_pct
            ):
                best_loss = current_loss
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1

            if no_improvement_iters >= self.early_stop_patience:
                logger.info(
                    "OptimizeExtrinsics: stopping early at iteration "
                    f"{i + 1} after {self.early_stop_patience} iterations "
                    "without improving by more than "
                    f"{self.early_stop_rel_improvement_pct}%"
                )
                break

        # Adam fallback: if L-BFGS left design vars as NaN, try Adam from last good state
        if not torch.isfinite(self.t).all() or not torch.isfinite(self.quat).all():
            logger.warning(
                "OptimizeExtrinsics: L-BFGS produced NaN output, "
                "falling back to Adam optimizer"
            )
            restore_state(last_good_state)
            adam_opt = torch.optim.Adam(self.design_vars, lr=1e-3)
            for _ in range(100):
                adam_opt.zero_grad()
                loss_adam = self.objective_function()
                if not torch.isfinite(loss_adam):
                    break
                loss_adam.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.design_vars, max_norm=1.0)
                adam_opt.step()
            self.loss = loss_adam.detach() if torch.isfinite(loss_adam) else self.loss

        # Hard fallback: if still NaN after Adam, use initial values (better than NaN)
        if not torch.isfinite(self.t).all() or not torch.isfinite(self.quat).all():
            logger.warning(
                "OptimizeExtrinsics: Adam fallback also failed, "
                "using initial (WHAM-derived) extrinsics values"
            )
            restore_state(initial_state)

        if self.printer:
            # Print the loss function components multiplied by their weights, if the weight term exists
            if "reprojection" in self.weights:
                print(
                    "reprojection loss: ",
                    self.weights["reprojection"]
                    * self.loss_reprojection(scale=self.reprojection_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if "quat_1" in self.weights:
                print(
                    "quat_1 loss: ",
                    self.weights["quat_1"] * self.loss_quat_1().detach().cpu().numpy(),
                )
            if "height" in self.weights:
                # print height error
                print(
                    "height error (m): ",
                    self.height - self.compute_smpl_height().detach().cpu().numpy(),
                )
                print(
                    "height loss: ",
                    self.weights["height"]
                    * self.loss_height(scale=self.height_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if "beta_change_regularization" in self.weights:
                print(
                    "beta_change_regularization loss: ",
                    self.weights["beta_change_regularization"]
                    * self.loss_beta_change_regularization().detach().cpu().numpy(),
                )

        output = {
            "t": self.t.detach(),
            "R": conversions.quaternion_to_rotation_matrix(self.quat).detach(),
            "beta": self.beta.detach(),
        }
        return output


class OptimizePose:

    def __init__(
        self,
        R_world_to_cam,
        t_world_to_cam,
        r_world_to_root,
        t_world_to_root,
        intrinsics,
        key2d_image,
        smpl_model,
        body_pose,
        beta,
        contact,
        frame_rate,
        optimize_camera=False,
        print_loss_terms=False,
        frame_range=None,
        weights=None,
        iterations=20,
        cutoff_frequency=12,
        smoothness_diff_n=1,
        output_dir=None,  # Optional: directory to save visualization plots
        video_path=None,  # Optional: path to video file for contact overlay
        frame_ids=None,  # Optional: frame IDs mapping video frames to contact data indices
        create_contact_visualizations=False,  # Optional: whether to create contact probability plots and video overlay
    ):
        """
        Initialize the pose optimizer.

        Args:
            R_world_to_cam: Camera rotation matrix (world to camera)
            t_world_to_cam: Camera translation vector (world to camera)
            r_world_to_root: Root orientation (in world coordinates)
            t_world_to_root: Root translation (in world coordinates)
            intrinsics: Camera intrinsic parameters
            key2d_image: 2D keypoints detected in image
            smpl_model: SMPL body model
            body_pose: Initial body pose parameters
            beta: SMPL body shape parameters
            contact: Contact labels for foot keypoints
            frame_rate: Frame rate of the sequence
            optimize_camera: Whether to optimize camera parameters
            print_loss_terms: Whether to print loss terms during optimization
            frame_range: Range of frames to optimize over
            weights: Weights for different loss terms
            iterations: Maximum number of optimization iterations
            cutoff_frequency: Cutoff frequency for frequency-domain smoothing
            smoothness_diff_n: Order of differentiation for smoothness term
        """

        self.device = R_world_to_cam.device

        if frame_range is None:
            frame_range = range(t_world_to_cam.shape[0])
            # minus 1 because of 0 indexing
            frame_range = range(frame_range[0], frame_range[-1] + 1)
        self.n_frames = len(frame_range)

        self.iterations = iterations
        self.conv_tol = 0.001
        self.early_stop_patience = 5
        self.early_stop_rel_improvement_pct = 1.0

        if frame_rate is None:
            self.frame_rate = 30
        else:
            self.frame_rate = frame_rate

        self.output_dir = output_dir
        self.video_path = video_path
        self.frame_ids = frame_ids
        self.create_contact_visualizations = create_contact_visualizations

        self.key2d_image = key2d_image[frame_range, :, :].unsqueeze(0)
        self.cam_center = torch.tensor(
            [intrinsics["cx"], intrinsics["cy"]], dtype=torch.float32, device=self.device
        ).reshape(1, 2)
        self.cam_f = torch.tensor(
            [intrinsics["fx"], intrinsics["fy"]], dtype=torch.float32, device=self.device
        ).reshape(1, 2)
        self.smpl_model = smpl_model
        self.beta = beta

        # distortion
        if "k1" in intrinsics:
            self.cam_distortion_k = (
                torch.tensor(
                    [intrinsics["k1"], intrinsics["k2"], intrinsics["k3"]],
                    dtype=torch.float32,
                    device=self.device,
                )
                .reshape(1, 3)
                .repeat(self.n_frames, 1)
            )
        else:
            self.cam_distortion_k = None
        if "p1" in intrinsics:
            self.cam_distortion_p = (
                torch.tensor(
                    [intrinsics["p1"], intrinsics["p2"]],
                    dtype=torch.float32,
                    device=self.device,
                )
                .reshape(1, 2)
                .repeat(self.n_frames, 1)
            )
        else:
            self.cam_distortion_p = None

        self.optimize_camera = optimize_camera
        self.print_loss_terms = print_loss_terms
        self.cutoff_frequency = cutoff_frequency
        self.n_frames = len(frame_range)
        self.smoothness_diff_n = torch.tensor(smoothness_diff_n)

        # These are used for contact losses; keep them ordered like this, it corresponds with
        # ordering in self.contact
        # Original: self.foot_names = ["LBigToe", "LHeel", "RBigToe", "RHeel", "LSmallToe", "RSmallToe"]
        # Now including small toes by copying big toe values
        self.foot_names = [
            "LBigToe",
            "LHeel",
            "RBigToe",
            "RHeel",
            "LSmallToe",
            "RSmallToe",
        ]
        self.foot_keypoints = [
            OPENPOSE_VERTICES_NAME.index(foot) for foot in self.foot_names
        ]

        # Expand contact matrix to include small toes (copy big toe values)
        # Original contact shape: (frames, 4) -> New shape: (frames, 6)
        original_contact = (
            contact[frame_range, :].unsqueeze(0).unsqueeze(-1)
        )  # Shape: (1, frames, 4, 1)

        # Copy LBigToe values for LSmallToe (index 0 -> index 4)
        # Copy RBigToe values for RSmallToe (index 2 -> index 5)
        left_small_toe_contact = original_contact[:, :, 0:1, :]  # Copy LBigToe
        right_small_toe_contact = original_contact[:, :, 2:3, :]  # Copy RBigToe

        # Concatenate to create expanded contact matrix
        self.contact = torch.cat(
            [
                original_contact,  # Original 4 contact points
                left_small_toe_contact,  # LSmallToe (copied from LBigToe)
                right_small_toe_contact,  # RSmallToe (copied from RBigToe)
            ],
            dim=2,
        )  # Shape: (1, frames, 6, 1)

        if weights is None:
            self.weights = {
                "reprojection": 10,
                "contact_velocity": 1,
                "contact_position": 100,
                #'frequency': 0, # seems to cause edge effects
                "smoothness_diff": 10,
                "flat_floor": 0,
            }

        else:
            self.weights = weights

        # design vars
        self.design_vars = []

        self.r_world_to_root = r_world_to_root[frame_range, :]
        self.r_world_to_root.requires_grad = True
        self.design_vars.append(self.r_world_to_root)

        self.t_world_to_root = t_world_to_root[frame_range, :]
        self.t_world_to_root.requires_grad = True
        self.design_vars.append(self.t_world_to_root)

        self.body_pose = body_pose[frame_range, :]
        self.body_pose.requires_grad = True
        self.design_vars.append(self.body_pose)

        # optionally optimize camera
        self.R_world_to_cam = R_world_to_cam[frame_range, :].unsqueeze(0)
        self.t_world_to_cam = t_world_to_cam[frame_range, :].unsqueeze(0)
        if self.optimize_camera:
            self.t = torch.mean(self.t_world_to_cam, axis=1).squeeze()  # fixed camera
            self.t_init = self.t.clone()
            self.t.requires_grad = True
            self.design_vars.append(self.t)

            self.quat = torch.mean(
                conversions.rotation_matrix_to_quaternion(
                    R_world_to_cam[frame_range, :, :]
                ),
                axis=0,
            )  # fixed camera
            self.quat_init = self.quat.clone()
            self.quat.requires_grad = True
            self.design_vars.append(self.quat)

        # Precomputed values (must be at end of init)
        if "reprojection" in self.weights and self.weights["reprojection"] > 0:
            self.confidence_selected_keypoints = (
                self.key2d_image[:, :, keypoints_to_use, 2]
                .unsqueeze(-1)
                .repeat(1, 1, 1, 2)
            )
            self.key2d_image_selected_keypoints = self.key2d_image[
                :, :, keypoints_to_use, :2
            ]
            self.forward_smpl()
            self.reprojection_loss_init = self.loss_reprojection().clone().detach()
            if not torch.isfinite(self.reprojection_loss_init) or self.reprojection_loss_init.item() == 0:
                logger.warning(
                    "reprojection_loss_init is inf/NaN/zero — using fallback scale."
                )
                self.reprojection_loss_init = torch.tensor(1.0, device=self.device)

        # if 'quat_1' in self.weights and self.weights['quat_1'] > 0:
        # self.quat_1_loss_init = self.loss_quat_1(scale=1).clone().detach()

        if (
            "camera_similarity" in self.weights
            and self.weights["camera_similarity"] > 0
        ):
            # it is hard to compute a scale factor for this b/c initial value will be 0.
            # instead perturb the initial camera parameters by 10% and see how much the loss changes

            # perturb self.quat by 10%
            # self.quat = self.quat * 1.1
            # self.t = self.t * 1.1
            # self.camera_similarity_loss_init = self.loss_camera_similarity().clone().detach()
            # reset the values
            # self.quat = self.quat / 1.1
            # self.t = self.t / 1.1
            self.camera_similarity_loss_init = torch.tensor(
                0.01, device=self.device
            )  # this caused errors...todo if implement

        if "pose_similarity" in self.weights and self.weights["pose_similarity"] > 0:
            # it is hard to compute a scale factor for this b/c initial value will be 0.
            # instead perturb the initial pose by 10% and see how much the loss changes
            self.pose_init = self.body_pose.clone()
            self.body_pose = self.body_pose * 1.1
            self.pose_similarity_loss_init = (
                self.loss_pose_similarity().clone().detach()
            )
            # reset the values
            self.body_pose = self.body_pose / 1.1

        if "root_translation_similarity" in self.weights and self.weights["root_translation_similarity"] > 0:
            # Normalize by initial value so the loss starts at 1.0, like all other terms.
            # The optimizer will reduce it from 1.0 toward 0.0.
            self.root_translation_similarity_loss_init = (
                self.loss_root_translation_similarity().clone().detach()
            )
            if not torch.isfinite(self.root_translation_similarity_loss_init) or self.root_translation_similarity_loss_init.item() == 0:
                self.root_translation_similarity_loss_init = torch.tensor(1.0, device=self.device)

        # Optimize debouncing: only call debounced_threshold once on original 4-point contact
        original_contact_for_debouncing = contact[frame_range, :]  # Shape: (frames, 4)

        if (
            "contact_position" in self.weights and self.weights["contact_position"] > 0
        ) or ("flat_floor" in self.weights and self.weights["flat_floor"] > 0):
            # Debounce the original 4-point contact matrix once
            original_contact_mask = self.debounced_threshold(
                original_contact_for_debouncing
            )

        if "contact_position" in self.weights and self.weights["contact_position"] > 0:
            # Expand the debounced mask to 6 points (copy big toe values for small toes)
            expanded_contact_mask = torch.cat(
                [
                    original_contact_mask,  # Original 4 contact points: LBigToe, LHeel, RBigToe, RHeel
                    original_contact_mask[:, 0:1],  # Copy LBigToe for LSmallToe
                    original_contact_mask[:, 2:3],  # Copy RBigToe for RSmallToe
                ],
                dim=1,
            )  # Shape: (frames, 6)

            self.contact_mask = expanded_contact_mask
            padded_mask = torch.cat(
                [
                    torch.zeros(1, self.contact_mask.shape[1], dtype=torch.bool, device=self.device),
                    self.contact_mask,
                    torch.zeros(1, self.contact_mask.shape[1], dtype=torch.bool, device=self.device),
                ]
            )
            self.contact_starts = (padded_mask[:-1] == False) & (
                padded_mask[1:] == True
            )
            self.contact_ends = (padded_mask[:-1] == True) & (padded_mask[1:] == False)
            self.contact_position_loss_init = (
                self.loss_contact_position().clone().detach()
            )

        if "contact_velocity" in self.weights and self.weights["contact_velocity"] > 0:
            self.contact_velocity_loss_init = (
                self.loss_contact_velocity().clone().detach()
            )
        if "frequency" in self.weights and self.weights["frequency"] > 0:
            self.loss_frequency_init = self.loss_frequency().clone().detach()
        if "smoothness_diff" in self.weights and self.weights["smoothness_diff"] > 0:
            self.smoothness_diff_loss_init = (
                self.loss_smoothness_diff().clone().detach()
            )
        if "flat_floor" in self.weights and self.weights["flat_floor"] > 0:
            # The small toe keypoints are anatomically higher than the heel and big toe keypoints on the SMPL model.
            # This offset is added to the small toe's y-position during the flat-floor loss calculation
            self.small_toe_offset = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.008, 0.008], device=self.device
            )

            # contact_mask may not have been built (e.g. treadmill where contact_position=0)
            if not hasattr(self, "contact_mask"):
                expanded_contact_mask = torch.cat(
                    [
                        original_contact_mask,
                        original_contact_mask[:, 0:1],  # LSmallToe <- LBigToe
                        original_contact_mask[:, 2:3],  # RSmallToe <- RBigToe
                    ],
                    dim=1,
                )
                self.contact_mask = expanded_contact_mask

            self.flat_floor_loss_init = self.loss_flat_floor().clone().detach()
        if "stability" in self.weights and self.weights["stability"] > 0:
            self.stability = ipman_ground_losses.StabilityLossCoP(
                self.smpl_model.bm.faces, device=self.device
            )
            self.stability_loss_init = self.loss_stability(scale=1).clone().detach()
        else:
            self.com = torch.zeros((self.n_frames, 3), device=self.device)

    def forward_smpl(self):
        """
        Forward pass through the SMPL model to compute 3D joints and vertices.

        Updates self.key3d (3D joint locations) and self.vertices (3D mesh vertices).
        """
        smpl_result = ut.pred_smpl(
            self.smpl_model,
            trans=self.t_world_to_root.unsqueeze(0),
            root_orient=self.r_world_to_root.unsqueeze(0),
            body_pose=self.body_pose.unsqueeze(0),
            betas=self.beta.unsqueeze(0),
        )
        """
        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        betas : B x D
        """
        self.key3d = smpl_result["joints3d_op"]
        self.vertices = smpl_result["verts3d_all"]

    def loss_reprojection(self, scale=1):
        """
        Calculate reprojection loss between 2D image keypoints and projected 3D SMPL keypoints.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Reprojection loss (scaled)
        """

        if self.optimize_camera:
            self.R_world_to_cam = (
                conversions.quaternion_to_rotation_matrix(self.quat)
                .unsqueeze(0)
                .repeat(self.n_frames, 1, 1)
                .unsqueeze(0)
            )
            self.t_world_to_cam = self.t.reshape(1, 1, 3)

        key2d_smpl = camera.reproject(
            self.key3d,
            self.R_world_to_cam,
            self.t_world_to_cam,
            self.cam_f,
            self.cam_center,
            cam_distortion_k=self.cam_distortion_k,
            cam_distortion_p=self.cam_distortion_p,
        )

        loss = (
            (
                self.confidence_selected_keypoints
                * (
                    key2d_smpl[:, :, keypoints_to_use, :]
                    - self.key2d_image_selected_keypoints
                )
            )
            ** 2
        ).sum()

        if self.print_loss_terms:
            print("reprojection loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_camera_similarity(self, scale=1):
        """
        Regularization term to keep camera parameters close to initial values.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Camera similarity loss (scaled)
        """
        if self.optimize_camera:
            quat_loss = (self.quat - self.quat_init) ** 2
            t_loss = (self.t - self.t_init) ** 2
            loss = quat_loss.sum() + t_loss.sum()
        else:
            loss = torch.tensor(0.01, device=self.device)

        if self.print_loss_terms:
            print("camera similarity loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_pose_similarity(self, scale=1):
        """
        Regularization term to keep pose parameters close to initial values.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Pose similarity loss (scaled)
        """
        body_pose_loss = (self.body_pose - self.pose_init) ** 2
        loss = body_pose_loss.sum()

        if self.print_loss_terms:
            print("pose similarity loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_quat_1(self, scale=1):
        """
        Constraint to ensure quaternion has unit norm.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Quaternion norm loss (scaled)
        """
        if self.optimize_camera:
            loss = (torch.norm(self.quat) - 1) ** 2
        else:
            loss = torch.tensor(
                0.01, device=self.device
            )  # will divide by this for the scale

        if self.print_loss_terms:
            print("quat norm loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_root_translation_similarity(self, scale=1):
        """
        For treadmill: penalize horizontal (x, z) displacement from the mean
        position so the model stays roughly stationary while allowing vertical
        (y) oscillation from the natural gait cycle.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Root horizontal position loss (scaled)
        """
        # t_world_to_root shape: (T, 3) — columns are x, y, z
        # Penalize deviation of x, z from their mean (stationary target)
        mean_xz = self.t_world_to_root[:, [0, 2]].mean(dim=0, keepdim=True)
        deviation = self.t_world_to_root[:, [0, 2]] - mean_xz
        loss = (deviation ** 2).sum()

        if self.print_loss_terms:
            print("root translation similarity loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_contact_velocity(self, scale=1):
        """
        Loss to enforce zero velocity at contact points (feet touching ground).

        Args:
            scale: Scaling factor for the loss

        Returns:
            Contact velocity loss (scaled)
        """
        # weight 0 velocity by contact probability
        # contact: L_toe, L_heel, R_toe, R_heel, L_small_toe, R_small_toe

        # velocity loss
        key3d_feet = self.key3d[:, :, self.foot_keypoints, :]
        speed_feet = self.compute_speed(key3d_feet, self.frame_rate)

        # filter self.contact with the same filter as the speed
        contact_loss = ((self.contact * speed_feet) ** 2).sum()

        if self.print_loss_terms:
            print(
                "contact velocity loss: ", (contact_loss / scale).detach().cpu().numpy()
            )

        return contact_loss / scale

    def loss_contact_position(self, scale=1):
        """
        Loss to enforce consistent foot position during contact phases.

        Calculates the variance of foot positions during each contact phase
        to ensure feet don't slide when in contact with the ground.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Contact position loss (scaled)
        """
        position_var_loss = 0

        key3d_feet = self.key3d[:, :, self.foot_keypoints, :]

        for n in range(len(self.foot_keypoints)):
            start_indices = torch.where(self.contact_starts[:, n])[0]
            end_indices = torch.where(self.contact_ends[:, n])[0]

            # get rid of any stretches that are too short (less than 2 frames, because we're looking at the variance - which requires at least 2 frames)
            diff_indices = abs(end_indices - start_indices)
            for i in range(len(diff_indices)):
                if diff_indices[i] < 2:
                    start_indices = torch.cat(
                        [start_indices[:i], start_indices[i + 1 :]]
                    )
                    end_indices = torch.cat([end_indices[:i], end_indices[i + 1 :]])

            # sum across directions of variance in each direction of foot positions in each contact stretch.
            # The position of the foot keypoint can change between contact phases, but should stay the same within one.
            # This should be more powerful than the velocity loss above for long standing activities.
            if not torch.isfinite(key3d_feet).all():
                logger.warning(
                    "Non-finite values (NaN or Inf) found in key3d_feet, returning zero loss for this term"
                )
                return torch.tensor(0.0, device=self.device)
            variances = [
                torch.var(key3d_feet[:, start:end, n, :], axis=1).sum()
                for start, end in zip(start_indices, end_indices)
                if end > start
            ]
            position_var_loss += (
                torch.sum(torch.stack(variances)) if variances else torch.tensor(0.0, device=self.device)
            )

        if self.print_loss_terms:
            print(
                "contact position loss:",
                (position_var_loss / scale).detach().cpu().numpy(),
            )

        return position_var_loss / scale

    def loss_flat_floor(self, scale=1):
        """
        Loss to enforce a flat floor by minimizing the variance in height (y-coordinate)
        of foot keypoints when they are in contact with the ground.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Flat floor loss (scaled)
        """
        key3d_feet_y = self.key3d[:, :, self.foot_keypoints, 1].squeeze()
        # Add an offset to the small toes to account for their anatomical position.
        # The offset is defined in __init__ for efficiency.
        key3d_feet_y_offset = key3d_feet_y + self.small_toe_offset
        masked_feet_y = key3d_feet_y_offset[self.contact_mask]

        # Calculate variance only if there are enough points for it to be meaningful
        if masked_feet_y.shape[0] > 1:
            loss = torch.var(masked_feet_y)
        else:
            loss = torch.tensor(0.0, device=self.device)

        if self.print_loss_terms:
            print("flat floor loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_frequency(self, scale=1):
        """
        Loss to minimize high-frequency components in the motion to produce smoother results.

        Applies a frequency-domain filter by penalizing power above a cutoff frequency.
        This helps remove jitter and noise from the motion while preserving natural movements.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Frequency-domain smoothing loss (scaled)
        """
        input_data = torch.cat(
            [self.body_pose, self.r_world_to_root, self.t_world_to_root], axis=1
        )

        # Compute FFT
        fft_result = torch.fft.rfft(
            input_data, dim=0
        )  # Real FFT along the time dimension

        # Calculate power spectrum (magnitude squared of FFT components)
        power_spectrum = torch.abs(fft_result) ** 2

        # Determine the frequency resolution
        frequency_resolution = self.frame_rate / input_data.size(0)

        # Find the index of the cutoff frequency
        cutoff_index = int(self.cutoff_frequency / frequency_resolution)

        # Calculate loss as the sum of power spectrum values above the cutoff frequency
        loss = power_spectrum[cutoff_index:].sum()

        if self.print_loss_terms:
            print("frequency loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_smoothness_diff(self, scale=1):
        """
        Loss to enforce smoothness by penalizing high derivatives of joint positions.

        Args:
            scale: Scaling factor for the loss
            diff_n: Order of differentiation (1 for velocity, 2 for acceleration)

        Returns:
            Smoothness loss (scaled)
        """
        # Speed diff of every position
        smoothness_diff = self.compute_speed(
            self.key3d, self.frame_rate, diff_n=self.smoothness_diff_n
        )

        # Calculate the loss as the sum of the squared velocity values
        loss = (smoothness_diff**2).sum()

        if self.print_loss_terms:
            print("smoothness diff loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def loss_stability(self, scale=1):
        """
        Loss to enforce physical stability by keeping the center of mass (COM)
        within the base of support polygon formed by the feet.

        Uses a soft constraint that penalizes COM positions outside the support polygon.

        Args:
            scale: Scaling factor for the loss

        Returns:
            Stability loss (scaled)
        """
        self.com = self.stability.compute_com(
            self.vertices.squeeze(0)
        )  # takes T,Nverts,3

        # x mediolateral, y up, z forward -> This will cause problems if y not up

        projected_com = self.com.clone()
        projected_com[..., 1] = 0  # zero-out y projects onto x-z plane

        # Base of support projection positions (T,nKeypoints,3)
        projected_foot_positions = (
            self.key3d[:, :, self.foot_keypoints_stability, :].squeeze(0).clone()
        )
        projected_foot_positions[..., 1] = 0  # zero-out y projects onto x-y plane

        # Step 1: Create 4 unit vectors pointing out from 4 polygon edges
        poly_vectors = projected_foot_positions - torch.roll(
            projected_foot_positions, shifts=1, dims=1
        )
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=projected_foot_positions.device)
        polygon_normal = torch.cross(
            poly_vectors, y_axis.expand_as(poly_vectors), dim=-1
        )
        polygon_normal = polygon_normal / torch.norm(
            polygon_normal, dim=-1, keepdim=True
        )  # Normalize

        # Step 2: Compute midpoints of polygon edges
        midpoints = (
            projected_foot_positions
            + torch.roll(projected_foot_positions, shifts=1, dims=1)
        ) / 2.0

        # Step 3: Vector from Midpoints to Projected COM
        com_vec_from_poly_edges = (
            projected_com.unsqueeze(1).repeat(1, len(self.foot_keypoints_stability), 1)
            - midpoints
        )

        # Step 4: Sum of Dot Products
        signed_distance_from_poly = torch.sum(
            com_vec_from_poly_edges * polygon_normal, dim=2
        )

        # Step 5: apply softplus (smooth relu): 0 inside polygon, >0 outside
        loss = (torch.nn.functional.softplus(signed_distance_from_poly) ** 2).sum()

        if self.print_loss_terms:
            print("stability loss:", (loss / scale).detach().cpu().numpy())

        return loss / scale

    def compute_speed(self, key_3d, frame_rate, diff_n=1):
        """
        Compute the velocity of points in a 3D trajectory.

        Args:
            key_3d: Tensor of shape (B, T, N, 3) containing 3D keypoint positions
            frame_rate: Frame rate of the data (frames per second)
            diff_n: Order of differentiation (1 for velocity, 2 for acceleration)

        Returns:
            Tensor of speeds (B, T, N, 1)
        """
        # Calculate the time interval between frames
        dt = 1.0 / frame_rate

        # Compute the difference in position between consecutive frames
        # We use the torch.diff function, which computes the discrete difference along a given axis (time axis in this case)
        position_diff = torch.diff(key_3d, dim=1, n=diff_n)

        # Compute velocity by dividing the position difference by the time interval
        velocity = position_diff / dt**diff_n
        # replicate the first time point to make the tensor the same size as the input
        velocity = torch.cat([velocity[:, 0, :, :].unsqueeze(1), velocity], dim=1)
        average_velocity = torch.norm(velocity, dim=-1, keepdim=True)

        return average_velocity

    def debounced_threshold(
        self,
        v_mask,
        # false_true_thresh_heel=0.38,
        # true_false_thresh_heel=0.65,
        # false_true_thresh_big_toe=0.5,
        # true_false_thresh_big_toe=0.45,
        false_true_thresh_heel=0.3,
        true_false_thresh_heel=0.75,
        false_true_thresh_big_toe=0.6,
        true_false_thresh_big_toe=0.38,
        min_stretch_len=1,
    ):
        """
        Apply a debounced threshold to a TxN matrix to reduce noise in contact detection.

        This implements a proper hysteresis state machine that considers the direction
        of movement to avoid rapid toggling between contact states.

        Args:
            v_mask: A TxN matrix of contact probabilities
            false_true_thresh_heel: Threshold for switching from False to True (no contact -> contact)
            true_false_thresh_heel: Threshold for switching from True to False (contact -> no contact)
            false_true_thresh_big_toe: Threshold for switching from False to True (no contact -> contact)
            true_false_thresh_big_toe: Threshold for switching from True to False (contact -> no contact)
            min_stretch_len: Minimum length of a stretch to trigger a state change

        Returns:
            Debounced binary contact mask (torch.bool)

        Hysteresis Logic:
            - When state is False: only switch to True if value goes ABOVE false_true_thresh
            - When state is True: only switch to False if value goes BELOW true_false_thresh
            - Values between thresholds maintain current state (no switching)
        """
        logger.info(
            "Debouncing contact mask with hysteresis logic using the following parameters:"
        )
        logger.info(f"false_true_thresh_heel: {false_true_thresh_heel}")
        logger.info(f"true_false_thresh_heel: {true_false_thresh_heel}")
        logger.info(f"false_true_thresh_big_toe: {false_true_thresh_big_toe}")
        logger.info(f"true_false_thresh_big_toe: {true_false_thresh_big_toe}")
        logger.info(f"min_stretch_len: {min_stretch_len}")

        T, N = v_mask.shape
        debounced = torch.zeros_like(v_mask, dtype=torch.bool)

        # Updated to handle 6 contact points: LBigToe, LHeel, RBigToe, RHeel, LSmallToe, RSmallToe
        foot_names = [
            "Left Big Toe",
            "Left Heel",
            "Right Big Toe",
            "Right Heel",
            "Left Small Toe",
            "Right Small Toe",
        ]

        for n in range(N):
            # Handle case where we might have fewer contact points than expected
            if n < len(foot_names):
                foot_name = foot_names[n]
            else:
                # Default to big toe behavior for any additional contact points
                foot_name = "Left Big Toe"

            if "Toe" in foot_name:
                false_true_thresh = false_true_thresh_big_toe
                true_false_thresh = true_false_thresh_big_toe
            else:
                false_true_thresh = false_true_thresh_heel
                true_false_thresh = true_false_thresh_heel

            column = v_mask[:, n]
            # Initialize state based on first value
            # Use clear hysteresis logic even for initialization
            first_val = column[0]
            if first_val > false_true_thresh:
                current_state = True
            else:
                current_state = False

            # Counters for debouncing
            consecutive_frames_wanting_change = 0
            target_state = None

            for t in range(T):
                current_value = column[t]

                # Initialize is_increasing for the first frame
                if t == 0:
                    is_increasing = True  # Default for first frame
                else:
                    # check is the line of probability is increasing or decreasing over the last 5 frames
                    if t > 1:
                        minus = t - 5 if t - 5 > 0 else t - (t - 1)
                        last_5_values = column[minus:t]
                        # check if the last 5 values are increasing or decreasing
                        # print("t: ", t)
                        # print("current_value: ", current_value)
                        # print("last_5_values: ", last_5_values)
                        if torch.all(last_5_values < current_value):
                            is_increasing = True
                        else:
                            is_increasing = False
                    else:
                        # For t == 1, compare with previous value
                        is_increasing = current_value > column[t - 1]

                # Determine what state this value "wants" based on hysteresis
                if current_state == False:
                    # When False, only consider switching to True if above false_true_thresh
                    if current_value > false_true_thresh and is_increasing:
                        desired_state = True
                    else:
                        desired_state = False  # Stay False
                else:  # current_state == True
                    # When True, only consider switching to False if below true_false_thresh
                    if current_value < true_false_thresh and not is_increasing:
                        desired_state = False
                    else:
                        desired_state = True  # Stay True

                # Handle debouncing
                if desired_state != current_state:
                    # We want to change state
                    if target_state == desired_state:
                        # We're continuing to want the same state change
                        consecutive_frames_wanting_change += 1
                    else:
                        # We want a different state change than before
                        target_state = desired_state
                        consecutive_frames_wanting_change = 1

                    # Check if we've wanted this change long enough
                    if consecutive_frames_wanting_change >= min_stretch_len:
                        current_state = desired_state
                        consecutive_frames_wanting_change = 0
                        target_state = None
                else:
                    # We don't want to change state, reset counters
                    consecutive_frames_wanting_change = 0
                    target_state = None

                debounced[t, n] = current_state

        return debounced

    def objective_function(self):
        """
        Compute the full objective function combining all weighted loss terms.

        This function aggregates all the individual loss terms according to their
        weights to form the complete optimization objective.

        Returns:
            Combined loss value
        """
        loss = 0
        if "reprojection" in self.weights and self.weights["reprojection"] > 0:
            loss += self.weights["reprojection"] * self.loss_reprojection(
                scale=self.reprojection_loss_init
            )
        if (
            "camera_similarity" in self.weights
            and self.weights["camera_similarity"] > 0
        ):
            loss += self.weights["camera_similarity"] * self.loss_camera_similarity(
                scale=self.camera_similarity_loss_init
            )
        if "pose_similarity" in self.weights and self.weights["pose_similarity"] > 0:
            loss += self.weights["pose_similarity"] * self.loss_pose_similarity(
                scale=self.pose_similarity_loss_init
            )
        if "quat_1" in self.weights and self.weights["quat_1"] > 0:
            loss += self.weights["quat_1"] * self.loss_quat_1()
        if "contact_position" in self.weights and self.weights["contact_position"] > 0:
            loss += self.weights["contact_position"] * self.loss_contact_position(
                scale=self.contact_position_loss_init
            )
        if "contact_velocity" in self.weights and self.weights["contact_velocity"] > 0:
            loss += self.weights["contact_velocity"] * self.loss_contact_velocity(
                scale=self.contact_velocity_loss_init
            )
        if "frequency" in self.weights and self.weights["frequency"] > 0:
            loss += self.weights["frequency"] * self.loss_frequency(
                scale=self.loss_frequency_init
            )
        if "smoothness_diff" in self.weights and self.weights["smoothness_diff"] > 0:
            loss += self.weights["smoothness_diff"] * self.loss_smoothness_diff(
                scale=self.smoothness_diff_loss_init
            )
        if "flat_floor" in self.weights and self.weights["flat_floor"] > 0:
            loss += self.weights["flat_floor"] * self.loss_flat_floor(
                scale=self.flat_floor_loss_init
            )
        if "stability" in self.weights and self.weights["stability"] > 0:
            loss += self.weights["stability"] * self.loss_stability(
                scale=self.stability_loss_init
            )
        if "root_translation_similarity" in self.weights and self.weights["root_translation_similarity"] > 0:
            loss += self.weights["root_translation_similarity"] * self.loss_root_translation_similarity(
                scale=self.root_translation_similarity_loss_init
            )

        return loss

    def optimize(self):
        """
        Run the optimization process to find optimal pose parameters.

        Uses L-BFGS optimizer to minimize the objective function, which combines
        reprojection error with physical constraints.

        Returns:
            Dictionary containing the optimized parameters:
            - t_root_in_world: Root translation
            - r_root_in_world: Root orientation
            - body_pose: Body pose parameters
            - key_3d: 3D joint positions
            - objective_values: Loss values during optimization
            - com: Center of mass positions
            - r_world_to_cam: Camera rotation matrix
            - t_world_to_cam: Camera translation vector
        """
        # Create an L-BFGS optimizer
        optimizer = torch.optim.LBFGS(
            self.design_vars,
            lr=2,
            tolerance_change=self.conv_tol,
            line_search_fn="strong_wolfe",
        )

        # optimizer = torch.optim.Adam(self.design_vars)

        # Define the closure function that reevaluates the model
        closure_call_count = [0]  # Use list to allow modification in closure
        nan_recovery_count = [0]  # Track NaN recoveries

        # Save initial state for NaN recovery
        saved_state = {
            "r_world_to_root": self.r_world_to_root.clone().detach(),
            "t_world_to_root": self.t_world_to_root.clone().detach(),
            "body_pose": self.body_pose.clone().detach(),
        }
        if self.optimize_camera:
            saved_state["t"] = self.t.clone().detach()
            saved_state["quat"] = self.quat.clone().detach()

        last_good_state = {
            "r_world_to_root": self.r_world_to_root.clone().detach(),
            "t_world_to_root": self.t_world_to_root.clone().detach(),
            "body_pose": self.body_pose.clone().detach(),
        }
        if self.optimize_camera:
            last_good_state["t"] = self.t.clone().detach()
            last_good_state["quat"] = self.quat.clone().detach()

        def restore_state(state_dict):
            """Restore design variables from a saved state."""
            with torch.no_grad():
                self.r_world_to_root.copy_(state_dict["r_world_to_root"])
                self.t_world_to_root.copy_(state_dict["t_world_to_root"])
                self.body_pose.copy_(state_dict["body_pose"])
                if self.optimize_camera and "t" in state_dict:
                    self.t.copy_(state_dict["t"])
                    self.quat.copy_(state_dict["quat"])

        def save_good_state():
            """Save current state as last known good."""
            last_good_state["r_world_to_root"] = self.r_world_to_root.clone().detach()
            last_good_state["t_world_to_root"] = self.t_world_to_root.clone().detach()
            last_good_state["body_pose"] = self.body_pose.clone().detach()
            if self.optimize_camera:
                last_good_state["t"] = self.t.clone().detach()
                last_good_state["quat"] = self.quat.clone().detach()

        # Initialize self.loss to a valid value to avoid AttributeError if first closure fails
        self.loss = torch.tensor(1e6, device=self.device, requires_grad=True)

        def closure():
            closure_call_count[0] += 1
            if closure_call_count[0] == 1:
                logger.info("First closure call - computing forward pass and loss")

            # Check for NaN in design variables BEFORE forward pass
            vars_to_check = [
                ("r_world_to_root", self.r_world_to_root),
                ("t_world_to_root", self.t_world_to_root),
                ("body_pose", self.body_pose),
            ]
            if self.optimize_camera:
                vars_to_check.extend([("t", self.t), ("quat", self.quat)])

            has_nan = False
            for var_name, var in vars_to_check:
                if not torch.isfinite(var).all():
                    has_nan = True
                    logger.warning(
                        f"NaN/Inf detected in {var_name} at closure call {closure_call_count[0]}"
                    )
                    break

            if has_nan:
                nan_recovery_count[0] += 1
                if nan_recovery_count[0] <= 3:
                    logger.warning(
                        f"Restoring from last good state (recovery #{nan_recovery_count[0]})"
                    )
                    restore_state(last_good_state)
                else:
                    logger.warning(
                        f"Too many NaN recoveries ({nan_recovery_count[0]}), restoring to initial state"
                    )
                    restore_state(saved_state)
                self.loss = torch.tensor(1e6, device=self.device, requires_grad=True)
                return self.loss

            optimizer.zero_grad()
            self.forward_smpl()

            # Check for NaN in key3d after forward pass
            if not torch.isfinite(self.key3d).all():
                nan_recovery_count[0] += 1
                logger.warning(
                    f"NaN/Inf detected in key3d at closure call {closure_call_count[0]}. Restoring state."
                )
                restore_state(last_good_state)
                self.loss = torch.tensor(1e6, device=self.device, requires_grad=True)
                return self.loss

            if closure_call_count[0] == 1:
                logger.info("Forward SMPL complete, computing objective function")
            self.loss = self.objective_function()

            # Check for NaN in loss
            if not torch.isfinite(self.loss):
                nan_recovery_count[0] += 1
                logger.warning(
                    f"NaN/Inf loss detected at closure call {closure_call_count[0]}. Restoring state."
                )
                restore_state(last_good_state)
                self.loss = torch.tensor(1e6, device=self.device, requires_grad=True)
                return self.loss

            # Save this as the last good state
            save_good_state()

            if closure_call_count[0] == 1:
                logger.info(f"Objective function complete, loss={self.loss.item():.6f}")
            if self.print_loss_terms:
                print("loss: ", self.loss.detach().cpu().numpy())
            self.loss.backward(retain_graph=True)
            if closure_call_count[0] == 1:
                logger.info("Backward pass complete")
            return self.loss

        # Optimization loop
        objective_values = []
        logger.info(f"Starting optimization loop: {self.iterations} iterations")
        max_nan_recoveries = 10
        best_loss = None
        no_improvement_iters = 0
        for i in range(self.iterations):
            if i % 5 == 0:  # Log every 5 iterations
                logger.info(f"Optimization iteration {i}/{self.iterations}")

            # Check if too many NaN recoveries - stop early
            if nan_recovery_count[0] >= max_nan_recoveries:
                logger.warning(
                    f"Stopping optimization early at iteration {i} due to {nan_recovery_count[0]} NaN recoveries"
                )
                break

            optimizer.step(closure)
            current_loss = self.loss.clone().detach().cpu()
            objective_values.append(current_loss)
            self.last_loss = current_loss

            current_loss_value = float(current_loss)
            relative_improvement_pct = 0.0
            if best_loss is not None:
                relative_improvement_pct = (
                    (best_loss - current_loss_value) / max(abs(best_loss), 1e-12)
                ) * 100.0

            if (
                best_loss is None
                or relative_improvement_pct > self.early_stop_rel_improvement_pct
            ):
                best_loss = current_loss_value
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1

            if no_improvement_iters >= self.early_stop_patience:
                logger.info(
                    "Stopping optimization early at iteration "
                    f"{i + 1} after {self.early_stop_patience} iterations "
                    "without improving by more than "
                    f"{self.early_stop_rel_improvement_pct}%"
                )
                break

        logger.info(
            f"Optimization complete after {i+1 if 'i' in dir() else self.iterations} iterations, {nan_recovery_count[0]} NaN recoveries"
        )

        if objective_values:
            objective_values = torch.stack(objective_values).to(self.device)
        else:
            objective_values = torch.empty(0, device=self.device)

        if self.print_loss_terms:
            # Print the loss function components multiplied by their weights, if the weight term exists
            if "reprojection" in self.weights and self.weights["reprojection"] > 0:
                print(
                    "weighted reprojection loss: ",
                    self.weights["reprojection"]
                    * self.loss_reprojection(scale=self.reprojection_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if (
                "camera_similarity" in self.weights
                and self.weights["camera_similarity"] > 0
            ):
                print(
                    "weighted camera similarity loss: ",
                    self.weights["camera_similarity"]
                    * self.loss_camera_similarity(
                        scale=self.camera_similarity_loss_init
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if (
                "pose_similarity" in self.weights
                and self.weights["pose_similarity"] > 0
            ):
                print(
                    "weighted pose similarity loss: ",
                    self.weights["pose_similarity"]
                    * self.loss_pose_similarity(scale=self.pose_similarity_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if (
                "contact_velocity" in self.weights
                and self.weights["contact_velocity"] > 0
            ):
                print(
                    "weighted contact velocity loss: ",
                    self.weights["contact_velocity"]
                    * self.loss_contact_velocity(scale=self.contact_velocity_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if (
                "contact_position" in self.weights
                and self.weights["contact_position"] > 0
            ):
                print(
                    "weighted contact position loss: ",
                    self.weights["contact_position"]
                    * self.loss_contact_position(scale=self.contact_position_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if "frequency" in self.weights and self.weights["frequency"] > 0:
                print(
                    "weighted frequency loss: ",
                    self.weights["frequency"]
                    * self.loss_frequency(scale=self.loss_frequency_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if (
                "smoothness_diff" in self.weights
                and self.weights["smoothness_diff"] > 0
            ):
                print(
                    "weighted smoothness diff loss: ",
                    self.weights["smoothness_diff"]
                    * self.loss_smoothness_diff(scale=self.smoothness_diff_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if "flat_floor" in self.weights and self.weights["flat_floor"] > 0:
                print(
                    "weighted flat floor loss: ",
                    self.weights["flat_floor"]
                    * self.loss_flat_floor(scale=self.flat_floor_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )
            if "stability" in self.weights and self.weights["stability"] > 0:
                print(
                    "weighted stability loss: ",
                    self.weights["stability"]
                    * self.loss_stability(scale=self.stability_loss_init)
                    .detach()
                    .cpu()
                    .numpy(),
                )

        output = {
            "t_root_in_world": self.t_world_to_root.detach(),
            "r_root_in_world": self.r_world_to_root.detach(),
            "body_pose": self.body_pose.detach(),
            "key_3d": self.key3d.detach(),
            "objective_values": objective_values,
            "com": self.com.detach(),
            "r_world_to_cam": self.R_world_to_cam.detach(),
            "t_world_to_cam": self.t_world_to_cam.detach(),
            "beta": self.beta.detach(),
        }
        return output
