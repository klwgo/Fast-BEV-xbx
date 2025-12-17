"""检查 synwoodscape subset pkl 中的关键信息（lidar2img / gt_drivable_mask）。"""
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='data/synwoodscape_infos_train_small.pkl',
                        help='需要检查的 pkl 路径')
    args = parser.parse_args()

    try:
        with open(args.pkl, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f'{args.pkl} 不存在')
        return

    infos = data.get('infos', [])
    if not infos:
        print(f'{args.pkl} 中 infos 为空')
        return

    info = infos[0]
    lidar2img = info.get('lidar2img')
    ann = info.get('ann_info', {})
    gt_drv = ann.get('gt_drivable_mask')

    print(f'文件: {args.pkl}')
    print(f'样本数: {len(infos)}')
    print(f'lidar2img 类型: {type(lidar2img)}, 是否 None: {lidar2img is None}')
    print(f'ann_info 键: {list(ann.keys())}')
    print(f'gt_drivable_mask 类型: {type(gt_drv)}, 是否 None: {gt_drv is None}')


if __name__ == '__main__':
    main()
