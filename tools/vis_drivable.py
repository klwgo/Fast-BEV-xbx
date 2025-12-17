# tools/vis_drivable.py
import os, torch, numpy as np, math
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.parallel import collate, scatter, DataContainer
from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataset
import torch.nn.functional as F

def main():
    cfg = Config.fromfile('configs/woodscape/fastbev_syn_headabc.py')
    cfg.data.val.ann_file = 'data/synwoodscape_infos_train_small_fixed.pkl'
    cfg.data.test = cfg.data.val
    cfg.data.samples_per_gpu = 1
    cfg.gpu_ids = [0]

    ckpt = 'work_dirs/fastbev_syn_headabc_small_overfit/latest.pth'
    device = torch.device('cuda:0')
    out_dir = 'work_dirs/vis_pred'; os.makedirs(out_dir, exist_ok=True)

    model = init_model(cfg, ckpt, device=device)
    # 强制每次重建 LUT，避免旧缓存尺寸不匹配
    if hasattr(model, 'fisheye_lut_cache') and model.fisheye_lut_cache is not None:
        model.fisheye_lut_cache.memory_cache.clear()
        if model.fisheye_lut_cache.cache_dir and model.fisheye_lut_cache.cache_dir.is_dir():
            for f in model.fisheye_lut_cache.cache_dir.glob('*.pt'):
                f.unlink(missing_ok=True)
        # 可视化阶段禁用磁盘缓存，确保 LUT 用当前参数重建
        model.fisheye_lut_cache.cache_dir = None
        model.fisheye_lut_cache.save_to_disk = False
    if hasattr(model, 'fisheye_force_rebuild'):
        model.fisheye_force_rebuild = True
    model.eval()
    # 需要 GT，使用训练 pipeline（包含 ann）构建数据
    data_cfg = cfg.data.train.copy()
    data_cfg.ann_file = 'data/synwoodscape_infos_train_small_fixed.pkl'
    dataset = build_dataset(data_cfg, dict(test_mode=False))

    saved = 0
    for idx in range(len(dataset)):
        if saved >= 5:
            break
        data = collate([dataset[idx]], samples_per_gpu=1)
        data = scatter(data, [0], dim=0)[0]

        # 兼容多视角展平：确保输入是 Tensor，最终形状 [B,V,3,H,W]
        def flatten_tensors(x):
            out = []
            if isinstance(x, torch.Tensor):
                out.append(x)
            elif hasattr(x, 'data'):  # DataContainer
                out.extend(flatten_tensors(x.data[0]))
            elif isinstance(x, list):
                for t in x:
                    out.extend(flatten_tensors(t))
            return out

        raw_img = data['img']
        tensors = flatten_tensors(raw_img)
        if not tensors:
            print(f"sample {idx}: img 无法转换为 Tensor，原始类型={type(raw_img)}, 内容示例={raw_img}")
            continue
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) > 1:
            print(f"sample {idx}: img shape 不一致 {shapes}，跳过")
            continue
        try:
            img_in = torch.stack(tensors, dim=0)
        except Exception as e:
            print(f"sample {idx}: stack 失败 {e}，跳过")
            continue

        if img_in.dim() == 6 and img_in.shape[:2] == (1, 1):  # [1,1,V,3,H,W]
            img_in = img_in.squeeze(0).squeeze(0)  # -> [V,3,H,W]

        if img_in.dim() == 5:
            pass  # 已是 [B,V,3,H,W]
        elif img_in.dim() == 4:
            if img_in.shape[1] == 3 and img_in.shape[0] > 1:  # [V,3,H,W]
                v, c, h, w = img_in.shape
                img_in = img_in.view(1, v, c, h, w)
            elif img_in.shape[1] % 3 == 0:  # [B, V*C, H, W]
                b, vch, h, w = img_in.shape
                num_views = vch // 3
                img_in = img_in.view(b, num_views, 3, h, w)
        elif img_in.dim() == 3:  # [V*C, H, W]
            vch, h, w = img_in.shape
            num_views = vch // 3
            img_in = img_in.view(1, num_views, 3, h, w)
        else:
            print(f"sample {idx}: 非法 img 形状 {img_in.shape}，跳过")
            continue
        print(f'sample {idx}: img_in shape after reshape {img_in.shape}')

        img_metas = data['img_metas']
        if hasattr(img_metas, 'data'):  # DataContainer
            img_metas = img_metas.data[0]
        # 转成列表传入 extract_feat
        if isinstance(img_metas, list):
            img_metas_list = img_metas
        else:
            img_metas_list = [img_metas]
        # 统一处理每个 meta，强制数值化
        cleaned_metas = []
        for m in img_metas_list:
            if not isinstance(m, dict):
                continue
            lidar2img = m.get('lidar2img')
            print(f'sample {idx}: lidar2img type={type(lidar2img)}')
            if isinstance(lidar2img, dict):
                if 'origin' not in lidar2img:
                    lidar2img['origin'] = np.zeros(3, dtype=np.float32)
                if isinstance(lidar2img.get('intrinsic'), list):
                    lidar2img['intrinsic'] = np.array(lidar2img['intrinsic'], dtype=np.float32)
                if isinstance(lidar2img.get('extrinsic'), list):
                    lidar2img['extrinsic'] = np.array(lidar2img['extrinsic'], dtype=np.float32)
            # img_shape 补充/压平
            if m.get('img_shape') is None and isinstance(m.get('cam_names'), list):
                h_list = []
                for cam_name in m['cam_names']:
                    cam = m.get(cam_name) or m.get('cams', {}).get(cam_name, {})
                    h = cam.get('height'); w = cam.get('width')
                    if h is not None and w is not None:
                        h_list.append((int(h), int(w), 3))
                if h_list:
                    m['img_shape'] = h_list
            if isinstance(m.get('img_shape'), list) and len(m['img_shape']) > 0:
                m['img_shape'] = m['img_shape'][0]
            cleaned_metas.append(m)
        img_metas_list = cleaned_metas
        if len(img_metas_list) == 0:
            print(f'sample {idx}: img_metas 非 dict，跳过'); continue
        if not isinstance(img_metas_list[0].get('lidar2img'), dict):
            print(f'sample {idx}: lidar2img 非 dict，跳过'); continue

        try:
            with torch.no_grad():
                # LUT 覆盖率统计（stride 取 1，对应原图分辨率）
                # get_points 是 fastbev 内部函数，用全路径引用
                from mmdet3d.models.detectors.fastbev import get_points
                voxel_size = torch.tensor(model.voxel_size[0])
                n_voxels = torch.tensor(model.n_voxels[0])
                origin = torch.tensor(img_metas_list[0]['lidar2img']['origin'])
                points = get_points(n_voxels=n_voxels, voxel_size=voxel_size, origin=origin).to(img_in.device)
                pixel_idx, valid_mask = model._get_fisheye_lut(
                    img_meta=img_metas_list[0],
                    points=points,
                    stride=1,
                    height=img_in.shape[-2],
                    width=img_in.shape[-1],
                    voxel_size=voxel_size.to(img_in.device),
                )
                print(f"sample {idx}: LUT valid={valid_mask.sum().item()} / {valid_mask.numel()}")

                feat_bev, valids, _ = model.extract_feat(img_in, img_metas_list, mode='test')
                # 某些风格返回 list，多尺度只取第一个尺度做检查
                if isinstance(feat_bev, (list, tuple)):
                    if len(feat_bev) == 0:
                        raise RuntimeError("feat_bev 为空")
                    feat_bev = feat_bev[0]
                if isinstance(valids, (list, tuple)):
                    valids = valids[0] if len(valids) > 0 else None
                preds = model.multitask_head(feat_bev)
            print(f'sample {idx}: feature_bev shape={feat_bev.shape}, mean={feat_bev.mean().item():.4f}, std={feat_bev.std().item():.4f}')
            if valids is not None and torch.is_tensor(valids):
                val_np = valids.detach().cpu().numpy()
                print(f'sample {idx}: valids shape={val_np.shape}, nonzero={int(val_np.sum())}')
            else:
                print(f'sample {idx}: valids is None')
        except Exception as e:
            import traceback
            print(f'sample {idx}: extract_feat failed with {e}\\n{traceback.format_exc()}, skip stats')
            preds = None

        pred = preds.get('drivable', None) if isinstance(preds, dict) else None
        if pred is None:
            print(f'sample {idx}: drivable missing'); continue
        pred = pred[0] if isinstance(pred, (list, tuple)) else pred
        if isinstance(pred, torch.Tensor):
            if pred.dim() == 3 and pred.size(0) > 1:
                pred = pred.softmax(0).cpu().numpy()[1]
            elif pred.dim() == 4 and pred.size(1) > 1:  # [B,C,H,W]
                pred = pred.softmax(1).cpu().numpy()[0, 1]
            else:
                pred = pred.sigmoid().cpu().numpy().squeeze()
        else:
            arr = np.array(pred)
            if arr.ndim == 3 and arr.shape[0] > 1:
                pred = torch.from_numpy(arr).softmax(0).cpu().numpy()[1]
            elif arr.ndim == 4 and arr.shape[1] > 1:
                pred = torch.from_numpy(arr).softmax(1).cpu().numpy()[0, 1]
            else:
                pred = arr.squeeze()

        # 取 GT
        def get_gt(d):
            def unwrap(x):
                if isinstance(x, DataContainer):
                    return unwrap(x.data)
                if isinstance(x, list) and len(x) == 1:
                    return unwrap(x[0])
                return x
            def load_if_path(x):
                x = unwrap(x)
                if isinstance(x, str) and os.path.exists(x):
                    if x.endswith('.npy'):
                        return np.load(x)
                return x
            for k in ['gt_drivable_mask', 'gt_bev_seg', 'gt_semantic_seg']:
                if k in d:
                    return load_if_path(d[k])
            ann = d.get('ann_info', {})
            if isinstance(ann, dict):
                for k in ['gt_drivable_mask', 'gt_bev_seg', 'gt_semantic_seg']:
                    if k in ann:
                        return load_if_path(ann[k])
            return None
        gt_raw = get_gt(data)
        if gt_raw is None:
            # 打印一次可用 key，便于排查
            print(f"sample {idx}: gt missing. data keys={list(data.keys())}, ann_info keys={list(data.get('ann_info',{}).keys()) if isinstance(data.get('ann_info',{}),dict) else 'N/A'}")
            continue
        if gt_raw is None:
            print(f'sample {idx}: gt missing'); continue
        gt = torch.tensor(gt_raw).float()
        if gt.dim() == 3:
            gt = gt[0]
        # 对齐到预测分辨率
        gt_resized = F.interpolate(gt[None, None].to(feat_bev.device),
                                   size=pred.shape[-2:], mode='nearest')[0, 0].cpu().numpy()

        # 保存可视化
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(gt_resized, cmap='gray'); axs[0].set_title(f'GT resized {idx}')
        axs[1].imshow(pred, cmap='viridis'); axs[1].set_title(f'Pred {idx}')
        for ax in axs: ax.axis('off')
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'vis_{idx}.png')
        plt.savefig(out_path, bbox_inches='tight'); plt.close(fig)
        print('saved', out_path)
        saved += 1

    print(f'done, saved {saved} images to {out_dir}')

if __name__ == '__main__':
    main()
