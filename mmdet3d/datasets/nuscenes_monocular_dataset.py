# -*- coding: utf-8 -*-
import torch
import numpy as np
import os

from mmdet.datasets import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .dataset_wrappers import MultiViewMixin
from IPython import embed
import mmcv
import skimage.io
import matplotlib.pyplot as plt
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import cv2
import imageio
from PIL import Image
import ipdb

import numpy as np, cv2
try:
    import imageio.v2 as imageio  # 需要 imageio-ffmpeg
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

def _make_even_wh(w, h):
    if w % 2: w += 1
    if h % 2: h += 1
    return w, h

def _sanitize_frames(frames, max_w=1920, max_h=1080, to_rgb=True):
    """统一帧尺寸/颜色：只缩小不放大、偶数宽高、所有帧同尺寸；可选 BGR->RGB。"""
    out, tgt_w, tgt_h = [], None, None
    for f in frames:
        if f is None: 
            continue
        g = f
        if to_rgb:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        h, w = g.shape[:2]
        scale = min(max_w/float(w), max_h/float(h), 1.0)
        nw, nh = _make_even_wh(int(w*scale), int(h*scale))
        if nw <= 0 or nh <= 0:
            nw, nh = 2, 2
        if (nw, nh) != (w, h):
            g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
        if tgt_w is None:
            tgt_w, tgt_h = nw, nh
        elif (nw, nh) != (tgt_w, tgt_h):
            g = cv2.resize(g, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        out.append(np.ascontiguousarray(g, dtype=np.uint8))
    # 帧数太少会触发“not enough frames”，复制一帧兜底
    if len(out) == 1:
        out.append(out[0].copy())
    return out

def _safe_write_video_rgb(frames_rgb, out_path, fps=10):
    """优先用 imageio 写 RGB；失败回退到 OpenCV（会转回 BGR）"""
    if HAS_IMAGEIO:
        try:
            imageio.mimsave(out_path, frames_rgb, fps=float(fps))
            return True
        except Exception:
            pass
    # 回退到 OpenCV：保证尺寸偶数
    bgr = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in frames_rgb]
    H, W = bgr[0].shape[:2]
    W, H = _make_even_wh(W, H)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    for fr in bgr:
        if fr.shape[1] != W or fr.shape[0] != H:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        vw.write(fr)
    vw.release()
    return True

def tofloat(x):
    return x.astype(np.float32) if x is not None else None

def _mosaic_imgs(imgs):
    n = len(imgs)
    if n == 0:
        return None
    # 统一高度（防止极端 resize 差异导致拼接报错）
    H = min(im.shape[0] for im in imgs if im is not None)
    W = min(im.shape[1] for im in imgs if im is not None)
    safe_imgs = [cv2.resize(im, (W, H)) for im in imgs if im is not None]

    if len(safe_imgs) == 0:
        return None

    if n == 6:
        top = np.concatenate(sort_list(safe_imgs[:3], [2,0,1]), axis=1)
        bot = np.concatenate(sort_list(safe_imgs[3:],   [2,0,1]), axis=1)
        return np.concatenate([top, bot], axis=0)

    if n == 4:
        row1 = np.concatenate(safe_imgs[0:2], axis=1)
        row2 = np.concatenate(safe_imgs[2:4], axis=1)
        return np.concatenate([row1, row2], axis=0)

    if n == 5:
        row1 = np.concatenate(safe_imgs[0:3], axis=1)
        row2 = np.concatenate(safe_imgs[3:5], axis=1)
        # 右侧补零到与 row1 同宽
        pad_w = row1.shape[1] - row2.shape[1]
        if pad_w > 0:
            row2 = np.pad(row2, ((0,0),(0,pad_w),(0,0)), mode='constant', constant_values=0)
        return np.concatenate([row1, row2], axis=0)

    # 兜底：全部横着拼
    return np.concatenate(safe_imgs, axis=1)

@DATASETS.register_module()
class NuScenesMultiViewDataset(MultiViewMixin, NuScenesDataset):
    def get_data_info(self, index):
        data_info = super().get_data_info(index)
        n_cameras = len(data_info['img_filename'])
        if not self.sequential and n_cameras != 6:
            try:
                import warnings
                warnings.warn(f"[Fast-BEV] Expected 6 cameras, got {n_cameras}. Continue with adaptive visualization.")
            except Exception:
                print(f"[Fast-BEV] Expected 6 cameras, got {n_cameras}. Continue with adaptive visualization.")

        new_info = dict(
            sample_idx=data_info['sample_idx'],
            img_prefix=[None] * n_cameras,
            img_info=[dict(filename=x) for x in data_info['img_filename']],
            lidar2img=dict(
                extrinsic=[tofloat(x) for x in data_info['lidar2img']],
                intrinsic=np.eye(4, dtype=np.float32),
                lidar2img_aug=data_info['lidar2img_aug'],
                lidar2img_extra=data_info['lidar2img_extra']
            )
        )
        if 'ann_info' in data_info:
            gt_bboxes_3d = data_info['ann_info']['gt_bboxes_3d']
            gt_labels_3d = data_info['ann_info']['gt_labels_3d'].copy()
            mask = gt_labels_3d >= 0
            gt_bboxes_3d = gt_bboxes_3d[mask]
            gt_names = data_info['ann_info']['gt_names'][mask]
            gt_labels_3d = gt_labels_3d[mask]
            new_info['ann_info'] = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_names=gt_names,
                gt_labels_3d=gt_labels_3d
            )
        return new_info

    def evaluate(self, results, *args, **kwargs):
        # update boxes with zero velocity
        new_results = []
        for i in range(len(results)):
            box_type = type(results[i]['boxes_3d'])
            boxes_3d = results[i]['boxes_3d'].tensor
            boxes_3d = box_type(boxes_3d, box_dim=9, origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
    
            new_results.append(dict(
                boxes_3d=boxes_3d,
                scores_3d=results[i]['scores_3d'],
                labels_3d=results[i]['labels_3d']
            ))
        
        vis_mode = kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        if vis_mode:
            embed(header='### vis nus test data ###')
            print('### vis nus test data ###')
            self.show(new_results, 'trash/test', thr=0.3)
            print('### finish vis ###')
            exit()
            
        if 'vis_mode' in kwargs.keys():
            kwargs.pop('vis_mode')
        
        result_dict = super().evaluate(new_results, *args, **kwargs)
        print(result_dict)
        return result_dict
    
    @staticmethod
    def draw_corners(img, corners, color, projection):
        corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
        corners_2d_3 = corners_3d_4 @ projection.T
        z_mask = corners_2d_3[:, 2] > 0
        corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
        corners_2d = corners_2d.astype(np.int)
        for i, j in [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]:
            if z_mask[i] and z_mask[j]:
                img = cv2.line(
                    img=img,
                    pt1=tuple(corners_2d[i]),
                    pt2=tuple(corners_2d[j]),
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)
        # drax `X' in the front
        if z_mask[0] and z_mask[5]:
            img = cv2.line(
                img=img,
                pt1=tuple(corners_2d[0]),
                pt2=tuple(corners_2d[5]),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)
        if z_mask[1] and z_mask[4]:
            img = cv2.line(
                img=img,
                pt1=tuple(corners_2d[1]),
                pt2=tuple(corners_2d[4]),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)

    def draw_bev_bbox_corner(self, img, box, color, scale_fac):
        box = box[:, None, :]  # [4,1,2]
        box = box + 50
        box = box * scale_fac
        box = np.int0(box)
        img = cv2.polylines(img, [box], isClosed=True, color=color, thickness=2)
        return img
    

    def show(self, results, out_dir='trash', bev_seg_results=None, thr=0.3, fps=3):
        assert out_dir is not None, 'Expect out_dir, got none.'
        colors = get_colors()
        all_img_gt, all_img_pred, all_bev_gt, all_bev_pred = [], [], [], []
        for i, result in enumerate(results):
            info = self.get_data_info(i)
            gt_bboxes = self.get_ann_info(i)
            print('saving image {}/{} to {}'.format(i, len(results), out_dir))
            # draw 3d box in BEV
            scale_fac = 10
            out_file_dir = str(i)
            ###### draw BEV pred ######
            bev_pred_img = np.zeros((100*scale_fac, 100*scale_fac, 3))
            if bev_seg_results is not None:
                bev_pred_road, bev_pred_lane = bev_seg_results[i]['seg_pred_road'], bev_seg_results[i]['seg_pred_lane']
                bev_pred_img = map2lssmap(bev_pred_road, bev_pred_lane)
                bev_pred_img = mmcv.imresize(bev_pred_img,
                                             (100*scale_fac, 100*scale_fac),
                                             interpolation='bilinear')
                
            scores = result['scores_3d'].numpy()
            try:
                bev_box_pred = result['boxes_3d'].corners.numpy()[:, [0, 2, 6, 4]][..., :2][scores > thr]
                labels = result['labels_3d'].numpy()[scores > thr]
                assert bev_box_pred.shape[0] == labels.shape[0]
                for idx in range(len(labels)):
                    bev_pred_img = self.draw_bev_bbox_corner(bev_pred_img, bev_box_pred[idx], colors[labels[idx]], scale_fac)
            except:
                pass
            
            bev_pred_img = process_bev_res_in_front(bev_pred_img)
            imsave(os.path.join(out_dir, out_file_dir, 'bev_pred.png'), mmcv.imrescale(bev_pred_img, 0.5))

            bev_gt_img = np.zeros((100*scale_fac, 100*scale_fac, 3))
            if bev_seg_results is not None:
                sample_token = self.get_data_info(i)['sample_idx']
                bev_seg_gt = self._get_map_by_sample_token(sample_token).astype('uint8')
                bev_gt_road, bev_gt_lane = bev_seg_gt[...,0], bev_seg_gt[...,1]
                bev_seg_gt = map2lssmap(bev_gt_road, bev_gt_lane)
                bev_gt_img = mmcv.imresize(
                    bev_seg_gt,
                    (100*scale_fac, 100*scale_fac),
                    interpolation='bilinear')
            try:
                # draw BEV GT
                bev_gt_bboxes = gt_bboxes['gt_bboxes_3d'].corners.numpy()[:,[0,2,6,4]][..., :2]
                labels_gt = gt_bboxes['gt_labels_3d']
                for idx in range(len(labels_gt)):
                    bev_gt_img = self.draw_bev_bbox_corner(bev_gt_img, bev_gt_bboxes[idx], colors[labels_gt[idx]], scale_fac)
            except:
                pass
            bev_gt_img = process_bev_res_in_front(bev_gt_img)
            imsave(os.path.join(out_dir, out_file_dir, 'bev_gt.png'), mmcv.imrescale(bev_gt_img, 0.5))
            all_bev_gt.append(mmcv.imrescale(bev_gt_img, 0.5))
            all_bev_pred.append(mmcv.imrescale(bev_pred_img, 0.5))
            ###### draw BEV pred ######
            ###### draw 3d box in image ######
            img_gt_list = []
            img_pred_list = []
            for j in range(len(info['img_info'])):
                img_pred = imread(info['img_info'][j]['filename'])
                img_gt = imread(info['img_info'][j]['filename'])
                # camera name
                camera_name = info['img_info'][j]['filename'].split('/')[-2]
                puttext(img_pred, camera_name)
                puttext(img_gt, camera_name)
                
                extrinsic = info['lidar2img']['extrinsic'][j]
                intrinsic = info['lidar2img']['intrinsic'][:3, :3]
                projection = intrinsic @ extrinsic[:3]
                if not len(result['scores_3d']):
                    pass
                else:
                    # draw pred
                    corners = result['boxes_3d'].corners.numpy()
                    scores = result['scores_3d'].numpy()
                    labels = result['labels_3d'].numpy()
                    for corner, score, label in zip(corners, scores, labels):
                        if score < thr:
                            continue
                        try:
                            self.draw_corners(img_pred, corner, colors[label], projection)
                        except:
                            pass
                    try:
                        # draw GT
                        corners = gt_bboxes['gt_bboxes_3d'].corners.numpy()
                        labels = gt_bboxes['gt_labels_3d']
                        for corner, label in zip(corners, labels):
                            self.draw_corners(img_gt, corner, colors[label], projection)
                    except:
                        pass
                out_file_dir = str(i)
                mmcv.mkdir_or_exist(os.path.join(out_dir, out_file_dir))
                # 缩小image大小 可视化方便一些
                img_gt_pred = np.concatenate([img_gt, img_pred], 0)
                imsave(os.path.join(out_dir, out_file_dir, '{}_gt_pred.png'.format(j)), mmcv.imrescale(img_gt_pred, 0.5))

                img_gt_list.append(mmcv.imrescale(img_gt, 0.5))
                img_pred_list.append(mmcv.imrescale(img_pred, 0.5))
            ###### draw 3d box in image ######
            ###### generate videos step:1 ######
            # tmp_img_up_pred = np.concatenate(sort_list(img_pred_list[0:3], sort=[2,0,1]), axis=1)
            # tmp_img_bottom_pred = np.concatenate(sort_list(img_pred_list[3:], sort=[2,0,1]) ,axis=1)
            # tmp_img_pred = np.concatenate([tmp_img_up_pred, tmp_img_bottom_pred], axis=0)
            # all_img_pred.append(tmp_img_pred)
            # tmp_img_up_gt = np.concatenate(sort_list(img_gt_list[0:3], sort=[2,0,1]),axis=1)
            # tmp_img_bottom_gt = np.concatenate(sort_list(img_gt_list[3:], sort=[2,0,1]),axis=1)
            # tmp_img_gt = np.concatenate([tmp_img_up_gt, tmp_img_bottom_gt], axis=0)
            # all_img_gt.append(tmp_img_gt)
            tmp_img_pred = _mosaic_imgs(img_pred_list)
            tmp_img_gt   = _mosaic_imgs(img_gt_list)

            if tmp_img_pred is not None:
                all_img_pred.append(tmp_img_pred)
            if tmp_img_gt is not None:
                all_img_gt.append(tmp_img_gt)

            ###### generate videos step:1 ######
        ###### generate videos step:2 ######
        gen_video(all_img_pred, all_bev_pred, out_dir, 'pred', fps=fps)
        gen_video(all_img_gt, all_bev_gt, out_dir, 'gt', fps=fps)
        ###### generate videos step:2 ######


# def sort_list(_list, sort):
#     assert len(_list) == len(sort)
#     new_list = []
#     for s in sort:
#         new_list.append(_list[s])
#     return new_list
def sort_list(_list, sort):
     """Safe sort: if length mismatches, return as-is."""
     try:
         if sort is None or len(_list) != len(sort):
             return _list
         return [_list[i] for i in sort]
     except Exception:
         # Any indexing error → fall back
         return _list


def get_colors():
    colors = np.multiply([
            plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
        ], 255).astype(np.uint8).tolist()
    colors = [i[::-1] for i in colors]
    return colors


def imread(img_path):
    img = mmcv.imread(img_path)
    return img


def imsave(img_path, img):
    mmcv.imwrite(img, img_path, auto_mkdir=True)


def puttext(img, name, loc=(30, 60), font=cv2.FONT_HERSHEY_DUPLEX ,color=(248, 202, 105)):
    try:
        cv2.putText(img, name, loc, font, 2, color, 2)
    except:
        img = Image.fromarray(img)
        img = np.array(img)
        cv2.putText(img, name, loc, font, 2, color, 2)

        
def map2lssmap(bev_map_road, bev_map_lane):
    white = [200, 200, 200]
    orange = [80, 127, 255]
    green = [171, 193, 115]
    bev_map = np.zeros_like(bev_map_road)[:,:,None].repeat(3,-1).astype('uint8')
    bev_map[bev_map[:, :, 0] == 0] = white
    bev_map[bev_map_road == 1] = orange
    bev_map[bev_map_lane == 1] = green
    return bev_map


def gen_video(img_list, bev_list, out_dir, mode='pred', fps=3):
    if mode == 'pred':
        out_video_path = os.path.join(out_dir, 'video_pred.mp4')
    else:
        assert mode == 'gt'
        out_video_path = os.path.join(out_dir, 'video_gt.mp4')
    tmp_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        bev = bev_list[i]
        bev = draw_ego_car(bev)
        bev = mmcv.imrescale(bev, img.shape[0]/bev.shape[0])
        # img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        # bev = cv2.copyMakeBorder(bev, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        tmp = np.concatenate([img, bev], axis=1)
        tmp = tmp[:, :, ::-1].astype('uint8')
        tmp_list.append(tmp)
        
    # imageio.mimsave(out_video_path, tmp_list, fps=fps)
    # 将帧统一到 ≤1920×1080、偶数宽高、相同尺寸，并转 RGB 供 imageio 写
    tmp_list_rgb = _sanitize_frames(tmp_list, max_w=1920, max_h=1080, to_rgb=True)
    assert len(tmp_list_rgb) > 0, "No frames to write"
    _safe_write_video_rgb(tmp_list_rgb, out_video_path, fps=fps)

    print('finish video generation')
    print('video path: {}'.format(out_video_path))


def process_bev_res_in_front(bev):
    bev = np.flip(bev, axis=0)
    return bev


def draw_ego_car(bev):
    h, w, _ = bev.shape
    ego_h = 8
    ego_w = 20
    x1 = (h - ego_h) // 2
    y1 = (w - ego_w) // 2
    x2 = (h + ego_h) // 2
    y2 = (w + ego_w) // 2
    color = [255, 0, 0]
    bev = cv2.rectangle(bev, (x1, y1), (x2, y2), color, -1)
    return bev
