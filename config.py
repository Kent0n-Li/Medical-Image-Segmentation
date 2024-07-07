import os
import sys
from multiprocessing import Lock

import argparse

colors = [
    (0, 0, 255),  # 红
    (0, 255, 0),  # 绿
    (255, 0, 0),  # 蓝
    (0, 255, 255),  # 黄
    (255, 0, 255),  # 紫
    (255, 255, 0),  # 青
    (192, 0, 0),  # 深红
    (0, 192, 0),  # 深绿
    (0, 0, 192),  # 深蓝
    (192, 192, 0),  # 橄榄
    (192, 0, 192),  # 紫罗兰
    (0, 192, 192),  # 蓝绿
    (128, 128, 128),  # 灰
    (128, 0, 0),  # 半深红
    (0, 128, 0),  # 半深绿
    (0, 0, 128),  # 半深蓝
    (128, 128, 0),  # 半深橄榄
    (128, 0, 128),  # 半深紫
    (0, 128, 128),  # 半深蓝绿
    (64, 128, 0),  # 橄榄/绿
    (128, 64, 0),  # 橄榄/红
    (64, 0, 128),  # 紫/蓝
    (128, 0, 64),  # 紫/红
    (0, 128, 64),  # 蓝绿/绿
    (0, 64, 128),  # 蓝绿/蓝
    (64, 0, 0),  # 暗红
    (0, 64, 0),  # 暗绿
    (0, 0, 64),  # 暗蓝
    (64, 64, 64),  # 暗灰
    (64, 64, 0),  # 暗橄榄
    (0, 64, 64),  # 暗蓝绿
    (64, 0, 64)  # 暗紫
]

global_logging_txt = "logging.txt"
output_file = 'command_output.txt'
validation_factor = 0.2

def parse_args(type=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=200, help='maximum epoch number to train')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    if type == 'train':
        parser.add_argument('--batch_size', type=int,
                            default=4, help='Batch Size')

    return parser.parse_args()


def print_web(text):
    with open(output_file, 'a') as f:
        f.write(text + '\n')