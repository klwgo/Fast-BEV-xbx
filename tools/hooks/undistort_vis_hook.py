# -*- coding: utf-8 -*-
"""训练过程中保存一帧鱼眼去畸变可视化."""

import os
import pickle

import cv2
import numpy as np
from mmcv.runner.hooks import HOOKS, Hook


def sanitize_distortion(raw_dist):
    if raw_dist is None:
        return np.zeros(4, dtype=np.float64)
    arr = np.asarray(raw_dist, dtype=np.float64).reshape(-1)
    if arr.size == 0 or float(np.max(np.abs(arr))) > 10.0:
        return np.zeros(4, dtype=np.float64)
    out = np.zeros(4, dtype=np.float64)
    out[:min(4, arr.size)] = arr[:4]
    return out


def undistort_fisheye(img, K, D):
    h, w = img.shape[:2]
    K = np.asarray(K, dtype=np.float64)
    D = sanitize_distortion(D).reshape(-1, 1)
    try:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3, dtype=np.float64), K, (w, h), cv2.CV_16SC2
        )
        undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    except Exception:
        undist = cv2.undistort(img, K, D)
    return undist


@HOOKS.register_module()
class UndistortVisHook(Hook):
    """每个 epoch 保存一帧去畸变前后对比图."""

    def __init__(self,
                 pkl_path,
                 cam='CAM_FRONT',
                 index=0,
                 out_dir='./work_dirs/undistort_vis'):
        self.pkl_path = pkl_path
        self.cam = cam
        self.index = index
        self.out_dir = out_dir
        with open(pkl_path, 'rb') as f:
            info = pickle.load(f)
        infos = info['infos']
        self.sample = infos[index]

    def after_train_epoch(self, runner):
        cam_info = self.sample['cams'][self.cam]
        img_path = cam_info['data_path']
        K = cam_info.get('cam_intrinsic')
        if K is None:
            K = cam_info.get('intrinsics')
        D = cam_info.get('cam_distortion')
        if D is None:
            D = cam_info.get('distortions')
        if D is None:
            D = [0, 0, 0, 0]
        img = cv2.imread(img_path)
        if img is None:
            runner.logger.warning(f'无法读取图像: {img_path}')
            return
        undist = undistort_fisheye(img, K, D)
        os.makedirs(self.out_dir, exist_ok=True)
        save_path = os.path.join(self.out_dir, f'epoch_{runner.epoch+1:03d}_{self.cam}.png')
        concat = np.concatenate([img, undist], axis=1)
        cv2.imwrite(save_path, concat)
        runner.logger.info(f'Saved undistort vis to {save_path}')
