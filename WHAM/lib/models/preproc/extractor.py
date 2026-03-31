from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import cv2
import torch
import numpy as np
import scipy.signal as signal
from progress.bar import Bar
from scipy.ndimage.filters import gaussian_filter1d

from .backbone.hmr2 import hmr2
from .backbone.utils import process_image
from ...utils.imutils import flip_kp, flip_bbox

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")

class FeatureExtractor(object):
    def __init__(self, device, flip_eval=False, max_batch_size=64):
        
        self.device = device
        self.flip_eval = flip_eval
        self.max_batch_size = max_batch_size
        
        ckpt = osp.join(ROOT_DIR, 'checkpoints', 'hmr2a.ckpt')
        self.model = hmr2(ckpt).to(device).eval()
    
    def run(self, video, tracking_results, patch_h=256, patch_w=256):
        
        if osp.isfile(video):
            cap = cv2.VideoCapture(video)
            is_video = True
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:   # Image list
            cap = video
            is_video = False
            length = len(video)
            height, width = cv2.imread(video[0]).shape[:2]
        
        # Get the frame_ids we need to process (only frames in tracking_results)
        frame_ids_to_process = set()
        for _id, val in tracking_results.items():
            if 'frame_id' in val:
                frame_ids_to_process.update(val['frame_id'])
        frame_ids_to_process = sorted(list(frame_ids_to_process))
        
        bar = Bar('Feature extraction ...', fill='#', max=len(frame_ids_to_process))
        
        for frame_id in frame_ids_to_process:
            if is_video:
                # Seek to the specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                flag, img = cap.read()
                if not flag:
                    continue
            else:
                if frame_id >= len(cap):
                    continue
                img = cv2.imread(cap[frame_id])
            
            for _id, val in tracking_results.items():
                if not frame_id in val['frame_id']: continue
                
                frame_id2 = np.where(val['frame_id'] == frame_id)[0][0]
                bbox = val['bbox'][frame_id2]
                cx, cy, scale = bbox
                
                norm_img, crop_img = process_image(img[..., ::-1], [cx, cy], scale, patch_h, patch_w)
                norm_img = torch.from_numpy(norm_img).unsqueeze(0).to(self.device)
                feature = self.model(norm_img, encode=True)
                tracking_results[_id]['features'].append(feature.cpu())
                
                if frame_id2 == 0: # First frame of this subject
                    tracking_results = self.predict_init(norm_img, tracking_results, _id, flip_eval=False)
                    
                if self.flip_eval:
                    flipped_bbox = flip_bbox(bbox, width, height)
                    # if tracking_results[_id]['flipped_bbox'] is not a key in tracking_results[_id], then set it to an empty list
                    if 'flipped_bbox' not in tracking_results[_id]:
                        tracking_results[_id]['flipped_bbox'] = []
                    tracking_results[_id]['flipped_bbox'].append(flipped_bbox)
                    
                    keypoints = val['keypoints'][frame_id2]
                    flipped_keypoints = flip_kp(keypoints, width)
                    if 'flipped_keypoints' not in tracking_results[_id]:
                        tracking_results[_id]['flipped_keypoints'] = []
                    tracking_results[_id]['flipped_keypoints'].append(flipped_keypoints)
                    
                    flipped_features = self.model(torch.flip(norm_img, (3, )), encode=True)
                    if 'flipped_features' not in tracking_results[_id]:
                        tracking_results[_id]['flipped_features'] = []
                    tracking_results[_id]['flipped_features'].append(flipped_features.cpu())
                    
                    if frame_id2 == 0:
                        tracking_results = self.predict_init(torch.flip(norm_img, (3, )), tracking_results, _id, flip_eval=True)
            
            bar.next()
        
        return self.process(tracking_results)
    
    def predict_init(self, norm_img, tracking_results, _id, flip_eval=False):
        prefix = 'flipped_' if flip_eval else ''
        
        pred_global_orient, pred_body_pose, pred_betas, _ = self.model(norm_img, encode=False)
        tracking_results[_id][prefix + 'init_global_orient'] = pred_global_orient.cpu()
        tracking_results[_id][prefix + 'init_body_pose'] = pred_body_pose.cpu()
        tracking_results[_id][prefix + 'init_betas'] = pred_betas.cpu()
        return tracking_results
    
    def process(self, tracking_results):
        output = defaultdict(dict)
        
        for _id, results in tracking_results.items():
            
            for key, val in results.items():
                if isinstance(val, list):
                    if isinstance(val[0], torch.Tensor):
                        val = torch.cat(val)
                    elif isinstance(val[0], np.ndarray):
                        val = np.array(val)
                output[_id][key] = val
        
        return output