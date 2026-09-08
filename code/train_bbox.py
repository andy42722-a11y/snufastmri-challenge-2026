import argparse
import os
import sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part_bbox import train as train_round2

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(
        description='Train VarNet bbox/round ablation arms on FastMRI challenge images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')

    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    parser.add_argument('--arm', type=str, default='A0_control')
    parser.add_argument('--bbox-weighted', action='store_true')
    parser.add_argument('--bbox-lambda', type=float, default=1.0)
    parser.add_argument('--edge-loss', action='store_true')
    parser.add_argument('--edge-beta', type=float, default=0.0)
    parser.add_argument('--lesion-sample', action='store_true')
    parser.add_argument('--lesion-repeat', type=int, default=2)
    parser.add_argument('--selection-metric', choices=['loss', 'bbox'], default='loss')

    parser.add_argument('--round2-progress', type=str, default='')
    parser.add_argument('--round2-results-csv', type=str, default='results/round2_results.csv')
    parser.add_argument('--round2-queued', type=str, default='')
    parser.add_argument('--round2-direction', type=str, default='')

    parser.add_argument('--round3-mode', action='store_true', help='Use Round 3 cosine schedule + val combined checkpoint selection')
    parser.add_argument('--round3-progress', type=str, default='')
    parser.add_argument('--round3-results-csv', type=str, default='results/round3_results.csv')
    parser.add_argument('--round3-val-curve', type=str, default='')
    parser.add_argument('--round3-queued', type=str, default='')
    parser.add_argument('--round3-direction', type=str, default='')
    parser.add_argument('--scheduler', choices=['none', 'cosine'], default='none')
    parser.add_argument('--min-lr', type=float, default=0.0)
    parser.add_argument('--selection-full-weight', type=float, default=0.5)
    parser.add_argument('--selection-bbox-weight', type=float, default=0.5)

    parser.add_argument('--round4-mode', action='store_true', help='Use Round 4 progress reporting with the cosine/combined trainer')
    parser.add_argument('--round4-progress', type=str, default='')
    parser.add_argument('--round4-results-csv', type=str, default='results/round4_results.csv')
    parser.add_argument('--round4-val-curve', type=str, default='')
    parser.add_argument('--round4-queued', type=str, default='')
    parser.add_argument('--round4-direction', type=str, default='')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    if args.round3_mode and args.round4_mode:
        raise ValueError('--round3-mode and --round4-mode are mutually exclusive')
    if args.round4_mode:
        # Existing trainer uses CosineAnnealingLR(T_max=args.num_epochs).
        args.round3_progress = args.round4_progress
        args.round3_results_csv = args.round4_results_csv
        args.round3_val_curve = args.round4_val_curve
        args.round3_queued = args.round4_queued
        args.round3_direction = args.round4_direction
        from round4_train_part import train as train_round4
        train_round4(args)
    elif args.round3_mode:
        from round3_train_part import train as train_round3
        train_round3(args)
    else:
        train_round2(args)
