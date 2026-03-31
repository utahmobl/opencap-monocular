import os
import argparse
import os.path as osp
import shutil

from collections import defaultdict
import copy
import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'WHAM')))

from configs.config import get_cfg_defaults
from utils.tracking_filters import (
    handle_multi_person_tracking,
    filter_frames_by_bbox_height,
    filter_frames_by_bbox_touching_edges,
    filter_frames_by_keypoints,
    InsufficientFullBodyKeypointsError,
)
from utils.utils_optim import get_openpose_keypoints
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

try:
    from utils.utils_optim import load_intrinsics
except ImportError:
    load_intrinsics = None

try:
    from lib.models.preproc.slam import SLAMModel

    _run_global = True
except:
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

initialized = False
cfg = None
network = None
smpl = None
detector = None
extractor = None

def initialize_wham():
    global initialized, cfg, network, smpl, detector, extractor
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not initialized:
        cfg = get_cfg_defaults()
        cfg.merge_from_file(os.path.join(repo_path, 'WHAM', 'configs', 'yamls', 'demo.yaml'))

        logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
        logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

        # ========= Load WHAM ========= #
        smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
        smpl = build_body_model(cfg.DEVICE, smpl_batch_size, gender="neutral")
        network = build_network(cfg, smpl)
        network.eval()

        vit_cfg = "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py"
        vit_ckpt = "checkpoints/vitpose+_huge_wholebody/wholebody.pth"

        detector = DetectionModel(
            cfg.DEVICE.lower(), vit_cfg=vit_cfg, vit_ckpt=vit_ckpt
        )

        extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)


        initialized = True

def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        intrinsics=None,
        run_global=False,
        save_pkl=False,
        visualize=False,
        run_smplify=False):

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Failed to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global

    # Preprocess
    with torch.no_grad():
        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and
                osp.exists(osp.join(output_pth, 'slam_results.pth'))):

            # 2D detection and SLAM
            global detector, extractor

            detector.initialize_tracking()
            bar = None

            if run_global:
                slam = SLAMModel(
                    video, output_pth, width, height, calib=calib, intrinsics=intrinsics
                )
                bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            else:
                slam = None
                bar = Bar('Preprocess: 2D detection', fill='#', max=length)
        
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break

                # 2D detection and tracking
                img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                detector.track(img_cv, fps, length)

                # SLAM
                if slam is not None:
                    slam.track()

                bar.next()

            tracking_results = detector.process(fps)

            filtering = True
            num_kpts = 133

            if filtering:
                def _n_tracked_frames(tr):
                    if not tr or 0 not in tr:
                        return 0
                    return len(tr[0]["frame_id"])

                logger.info(
                    f"[WHAM preprocess] video frames (cap)={length}, "
                    f"after detector.process={_n_tracked_frames(tracking_results)}"
                )

                # Handle multi-person detection and select person of interest
                tracking_results = handle_multi_person_tracking(
                    tracking_results, video, num_kpts, length, output_pth
                )

                # make sure the tracking results is a dictionary of length 1 with the key 0 (only one person of interest)
                if len(tracking_results) == 0:
                    raise ValueError("No tracking results found")
                if len(tracking_results) != 1 or list(tracking_results.keys())[0] != 0:
                    raise ValueError("Tracking results is not a dictionary with the key 0")

                logger.info(
                    f"[WHAM preprocess] after handle_multi_person: {_n_tracked_frames(tracking_results)} frames"
                )

                # Filter start and end frames based on bbox height
                tracking_results = filter_frames_by_bbox_height(tracking_results, height, width)
                logger.info(
                    f"[WHAM preprocess] after bbox_height filter: {_n_tracked_frames(tracking_results)} frames"
                )

                # Filter start and end frames where bbox touches the image edges
                tracking_results = filter_frames_by_bbox_touching_edges(tracking_results, height, width)
                logger.info(
                    f"[WHAM preprocess] after bbox_edges filter: {_n_tracked_frames(tracking_results)} frames"
                )

                # Filter the tracking results based on the number of visible keypoints with a confidence score greater than X
                tracking_results = filter_frames_by_keypoints(tracking_results)
                logger.info(
                    f"[WHAM preprocess] after keypoint_conf filter: {_n_tracked_frames(tracking_results)} frames"
                )

                length = len(tracking_results[0]['frame_id'])

            if slam is not None:
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0  # Unit quaternion


            # Extract image features (only frames in tracking_results)
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')

            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            logger.info(f'Save processed data at {output_pth}')

        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')


    tracking_results_reproj = copy.deepcopy(tracking_results)
    # keep only the first 17 keypoints of tracking_results['keypoints']
    for _id in tracking_results.keys():
        tracking_results[_id]['keypoints'] = tracking_results[_id]['keypoints'][:, :17]

    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)

    # run WHAM
    results = defaultdict(dict)

    n_subjs = len(dataset)
    # make sure there is only one subject
    if n_subjs != 1:
        raise ValueError("There should be only one subject in the dataset")
    
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                                       return_y_up=True, **kwargs)

                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                               return_y_up=True, **kwargs)

                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2

                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch

                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                               return_y_up=True, **kwargs)

        # if False:
        if run_smplify:
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            pred = smplify.fit(pred, input_keypoints, **kwargs)

            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()

        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id

        # additional results for opencap-mono
        results[_id]["contact"] = pred["contact"].cpu().squeeze(0).numpy()
        results[_id]["poses_body"] = pred["poses_body"].cpu().squeeze(0).numpy()
        results[_id]["poses_root_cam"] = pred["poses_root_cam"].cpu().squeeze(1).numpy()
        # results[_id]["betas"] = pred["betas"].cpu().squeeze(0).numpy()
        results[_id]["verts_cam"] = (
            (pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)).cpu().numpy()
        )
        results[_id]["poses_root_world"] = (
            pred["poses_root_world"].cpu().squeeze(0).numpy()
        )
        results[_id]["trans_world"] = pred["trans_world"].cpu().squeeze(0).numpy()
        results[_id]["trans_cam"] = pred["trans_cam"].cpu().squeeze(0).numpy()
        results[_id]["frame_id"] = frame_id
        results[_id]["cam_trans"] = pred["cam"].cpu().squeeze(0).numpy()
        results[_id]["tracking_results_for_reproj"] = tracking_results_reproj[_id]
        results[_id]["cam_R"] = pred["cam_R"].cpu().squeeze(0).numpy()
        results[_id]["cam_T"] = pred["cam_T"].cpu().squeeze(0).numpy()
        ##

    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))

    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)


def main_wham(
    video_path,
    output_path,
    calib_path,
    intrinsics_path=None,
    estimate_local_only=True,
    visualize=True,
    save_pkl=True,
    run_smplify=True,
    rerun=False,
):
    # get repo path
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(repo_path, 'WHAM', 'configs', 'yamls',  'demo.yaml'))

    # ========= Load WHAM ========= #
    # Initialize WHAM only once
    global initialized
    if not initialized:
        logger.info("Initialized: False")
        initialize_wham()
    else:
        logger.info("Initialized: True")
    global network, smpl

    # Output folder
    sequence = '.'.join(video_path.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(output_path, sequence)

    # Delete existing output folder if any
    if osp.exists(output_pth):
        # if we want to rerun this video, delete the existing output folder
        if rerun:
            shutil.rmtree(output_pth)
            os.makedirs(output_pth, exist_ok=True)
        # else skip this video
        else:
            logger.info(f'WHAM already ran on {video_path}.')
            return
    #  if not exists, create the output folder
    else:
        os.makedirs(output_pth, exist_ok=True)

    intrinsics = None
    if intrinsics_path and load_intrinsics and osp.isfile(intrinsics_path):
        intrinsics = load_intrinsics(intrinsics_path)
        logger.info(
            f"Loaded intrinsics from {intrinsics_path} "
            f"(fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f})"
        )
    if intrinsics is not None:
        calib_path = None

    try:
        run(
            cfg,
            video_path,
            output_pth,
            network,
            calib=calib_path,
            intrinsics=intrinsics,
            run_global=not estimate_local_only,
            save_pkl=save_pkl,
            visualize=visualize,
            run_smplify=run_smplify,
        )
    except InsufficientFullBodyKeypointsError:
        # Remove partial output so a later run without rerun=True does not skip WHAM.
        if osp.exists(output_pth):
            shutil.rmtree(output_pth)
            logger.info(f"Removed incomplete WHAM output after keypoint gate: {output_pth}")
        raise

    logger.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str,
                        default='../LabValidation_withVideos1/subject3/VideoData/Session1/Cam1/walking1/walking1.avi',
                        help='Input video path or YouTube link')

    parser.add_argument('--output_pth', type=str, default='../output/demo',
                        help='Output folder to write results')

    parser.add_argument('--calib', type=str, default='../examples/walking4/calib.txt',
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')

    parser.add_argument('--visualize', action='store_true',
                        default=True,
                        help='Visualize the output mesh if True')

    parser.add_argument('--save_pkl', action='store_true',
                        default=True,
                        help='Save output as pkl file')

    parser.add_argument('--run_smplify', action='store_true',
                        default=True,
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()

    main_wham(
        video_path=args.video,
        output_path=args.output_pth,
        calib_path=args.calib,
        estimate_local_only=args.estimate_local_only,
        visualize=args.visualize,
        save_pkl=args.save_pkl,
        run_smplify=args.run_smplify
    )
