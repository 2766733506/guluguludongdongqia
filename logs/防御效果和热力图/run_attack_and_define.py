import argparse
import os
import glob
import random
import time
from logs.防御效果和热力图.ig全自动检验 import auto_ig, auto_gifd
import numpy as np

import sys


def get_png_abs_paths(dir_path):
    dir_path = os.path.abspath(dir_path)
    png_files = glob.glob(os.path.join(dir_path, "*.png"))
    png_files = sorted(os.path.abspath(p) for p in png_files)
    return png_files


sys.path.append('.')
sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('/home/Suyilin/paper_code/grad_cam_plus/GIFD_Gradient_Inversion_Attack/')

from 热力图测试脚本全自动 import make_log


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0,
                        help='device to work on')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='要测试的图片的路径名称，e.g.sampled_images/cifar10_32')
    parser.add_argument('--basepath', type=str, required=True,
                        help='结果存放的根目录')
    parser.add_argument('--model_path', type=str, required=True,
                        help='使用模型的路径，是完整的模型，不是字典')

    parser.add_argument(
        '--class_name',
        type=str,
        choices=['cifar10', 'cifar100', 'tiny-imagenet-200'],
        required=True,
        help='数据集是什么'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=32,
        help='输入图像的宽和高，默认22'
    )
    parser.add_argument(
        '--epsilon',
        type=int,
        default=5,
        help='使用的隐私预算'
    )

    return parser.parse_args()


args = get_args()
device = 'cuda:' + str(args.gpu)  # 从命令行参数接收

image_path = get_png_abs_paths(args.image_folder)
basepath = args.basepath
model_path = args.model_path

for i in image_path:
    try:
        make_log(i, model_path, device=device, base_path=basepath, class_name=args.class_name, epsilon=args.epsilon,image_size=(args.image_size, args.image_size))
    except Exception as e:
        # 记录到 logger.txt（追加）
        with open("logger.txt", "a", encoding="utf-8") as f:
            f.write(f"i={i}\n")
            f.write(f"e={e}\n")
            f.write(f"\n")
        continue

for i in image_path:
    try:
        auto_gifd(i, model_path, device=device, base_path=basepath, class_name='args.class_name')
        auto_ig(i, model_path, device=device, base_path=basepath, class_name='args.class_name',img_shape=(3,args.image_size,args.image_size))
    except Exception as e:
        # 记录到 logger.txt（追加）
        with open("logger.txt", "a", encoding="utf-8") as f:
            f.write(f"i={i}\n")
            f.write(f"e={e}\n")
        continue



