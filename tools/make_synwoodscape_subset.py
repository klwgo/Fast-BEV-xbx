import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser(description='Make a small subset pkl for SynWoodScape.')
    parser.add_argument('--in-pkl', default='data/synwoodscape_infos_train.pkl', help='Input full pkl')
    parser.add_argument('--out-pkl', default='data/synwoodscape_infos_train_small.pkl', help='Output subset pkl')
    parser.add_argument('--num', type=int, default=100, help='Number of samples to keep')
    args = parser.parse_args()

    if not os.path.exists(args.in_pkl):
        raise FileNotFoundError(args.in_pkl)

    with open(args.in_pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data.get('infos', [])
    subset = infos[:args.num]
    data['infos'] = subset
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {len(subset)} infos to {args.out_pkl}')


if __name__ == '__main__':
    main()
