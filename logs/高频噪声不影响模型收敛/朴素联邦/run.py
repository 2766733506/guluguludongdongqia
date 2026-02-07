import os
import sys

import torch

sys.path.append('../../../')

# 整个模拟联邦学习的流程文件
import logging
from recovery import construct_model
from server import Server
from client import Client
from data import split_datasets_dirichlet, split_datasets_class_balanced
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='tip',
                        help='training or defense method')
    parser.add_argument(
        '--method',
        type=str,
        choices=['tip', 'dp', 'agp','normal'],
        required=True,
        help='使用什么方法')
    parser.add_argument('--gpu', type=int, default=0,
                        help='device to work on')

    return parser.parse_args()


# method = 'apg'
# work_device = 'cuda:0'
args = get_args()
method = args.method  # 从命令行参数接收
if 2 >= args.gpu >= 0:
    work_device = 'cuda:' + str(args.gpu)
else:
    work_device = 'cpu'

base_dir = '/home/Suyilin/paper_code/grad_cam_plus/logs/高频噪声不影响“模型收敛/朴素联邦/每一轮的模型'
dataset = 'cifar10'
save_base_path = f"{base_dir}/{method}_no-iid_0.5/{dataset}"
os.makedirs(save_base_path,exist_ok=True)

communicate_round = 1000
# 先初始化一个模型
global_model, _ = construct_model('ResNet18', 10)

# 创建1个真实的用户去模拟多用户
# datasets = split_datasets_dirichlet(100)
datasets = split_datasets_class_balanced(100)
true_client = Client(global_model, datasets, one_true_user_simulation_num=10, obj_num=100, device=work_device)


# 自定义logger
logger = logging.getLogger(f"{method}_logger")
logger.setLevel(logging.INFO)
# 创建文件处理器（每次都会添加）
# file_handler = logging.FileHandler(f"accuracy_{method}_no-IID_0.5.log", mode='a', encoding='utf-8')
file_handler = logging.FileHandler(f"accuracy_{method}.log", mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
# 创建格式器，只输出 message 内容
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
# 添加 handler 到 logger（每次运行都会重复添加）
logger.addHandler(file_handler)


# 自定义一个config生成器
def make_train_config(round: int):
    config = {
        'lr': 0.001,
        'epoch': 2,
        # 这个是隐私预算，直接对应的每一轮中每个用户的
        'epsilon': 5,
        'fit_method': method,
        'delta': 1e-5,
        'clip': 5,
        'tip_noise_beta': 0.1,
        # tip_noise_beta照此推导
        # 4 * batch * lr * (max_grad_norm / 1) * math.sqrt(2 * math.log(1.25 / 1e-5)) epsilon
        # 这里的batch是代表样本的batch，为1
        'noise_multiplier': 0.1
    }
    return config


server = Server(init_model=global_model.state_dict(),
                communication_round=communicate_round,
                client=true_client,
                logger=logger,
                args=args,
                config_generate=make_train_config,
                save_base_path=save_base_path,
                work_device=work_device,
                init_round_num=0
                )
server.start()
