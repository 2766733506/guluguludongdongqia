# 不用flwr的结构了，什么垃圾
# 自己写一个服务器框架，知识本地模拟
import os

import torch

from recovery import construct_model
from tool.fix_dp import make_resnet_dp_compatible
from tool.test_acc import test_cifar10_accuracy

save_path = r'/home/Suyilin/paper_code/grad_cam_plus/logs/高频噪声不影响“模型收敛/朴素联邦/每一轮的模型/normal/cifar10'

# 整个联邦学习的用户数量
N = 100
# 每轮选择多少客户端进行训练，此为比例
fraction_fit = 1.0


class Server:
    def __init__(self,init_model,   # 注意，这里是参数字典，模型结构维护于client中
                 communication_round,
                 client,
                 save_base_path,
                 logger,args,
                 datasets_type='cifar10',
                 config_generate=None,
                 model_name='ResNet18',
                 work_device='cuda:0',
                 class_num=10,
                 init_round_num=0,):
        self.model = init_model
        self.communication_round = communication_round
        self.client = client
        self.args = args
        self.save_base_path = save_base_path
        self.logger = logger
        self.datasets_type = datasets_type
        self.config_generate = config_generate
        self.model_name = model_name
        self.work_device = work_device
        self.class_num = class_num
        self.init_round_num = init_round_num



    def global_evaluation(self):

        if self.datasets_type == 'cifar10':
            temp, _ = construct_model(self.model_name,self.class_num)
            temp.load_state_dict(self.model)
            return test_cifar10_accuracy(temp,device=self.work_device)

    def start(self):
        for r in range(self.communication_round):
            print('round',r + self.init_round_num)
            # 首先是给客户端分发模型
            self.client.model.load_state_dict(self.model)
            # 唯一实例化的用户开始模拟
            config = self.config_generate(r)
            method = config['fit_method']
            model_dict, data_num, metric =self.client.fit(self.model,config)
            torch.save(model_dict, os.path.join(self.save_base_path,f'resnet18_cifar10_dict_{method}_round_{r + self.init_round_num}.pth'))
            acc = self.global_evaluation()
            self.model = model_dict
            print(f'round-{r + self.init_round_num},acc-{acc}')
            self.logger.info(f'round-{r + self.init_round_num},acc-{acc}')






