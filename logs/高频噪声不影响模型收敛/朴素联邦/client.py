import math
import random
import time
from copy import deepcopy

import numpy as np
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from tool.locla_cam import HookGradCam
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class RandomCycleSampler:
    def __init__(self, n: int):
        self.n = n
        self.pool = list(range(n))

    def sample(self) -> int:
        if not self.pool:
            # 一轮用完，重新初始化
            self.pool = list(range(self.n))
        idx = random.choice(self.pool)
        self.pool.remove(idx)
        return idx


class Client:
    def __init__(self, model, train_data_list, chanel_num=3, image_size=(32, 32), fit_method='normal',
                 criterion=CrossEntropyLoss(), device='cuda:0', one_true_user_simulation_num=1, obj_num=100):
        # 总共要模拟多少个用户，（划分了多少个子数据集）
        self.obj_num = obj_num
        self.data_num = None
        self.add_noise_loader = None
        self.train_data_loader = None
        self.train_data = None
        self.model = model
        # 这个变量表示的是，再资源受限制的情况下，用一个真实在线用户模拟多少个在线用户
        self.one_true_user_simulation_num = one_true_user_simulation_num
        # 保证数据全随机性覆盖
        self.simpler = RandomCycleSampler(obj_num)
        self.model_keys = list(model.state_dict().keys())
        # 这个train_data_list是被划分成n个用户的数据集，10个真实虚拟出来的用户模拟100个用户
        # 之后每个用户被采样之后，执行fit时，先随机一下自己的数据，以此达到模拟多用户
        self.train_data_list = train_data_list

        # 传进来的train——data——list是一个数据集列表，也是整个项目要模拟的用户数量
        self.num_train_data_list = len(train_data_list)

        # 本该初始化data——loader的，但是，经过求改，再fit函数一开始的时候自动生成

        self.fit_method = fit_method
        self.criterion = criterion
        self.device = device
        self.batchsize = 32

        # 这里两个data_loader是选择重要通道的时候使用的，无论是选所有样本还是特定样本，都可以在这里控制
        self.agp_add_noise_data_loader = None
        self.suyilin_add_noise_data_loader = None

        # 这里记录这个客户端所用数据的大小，通道
        self.chanel_num = chanel_num
        self.size = image_size

    def select_one_sample_per_class(self, dataset):
        label_to_index = dict()

        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label not in label_to_index:
                label_to_index[label] = idx
            # 如果每一类都找到了，就可以提前结束
            if len(label_to_index) == len(set(label_to_index.values())):
                break

        selected_indices = list(label_to_index.values())
        subset = Subset(dataset, selected_indices)

        return DataLoader(subset, batch_size=1, shuffle=True, num_workers=4)

    def aggregate(self, state_dicts_with_samples_list):
        """
        FedAvg 聚合函数
        输入：
            state_dicts_with_samples: List of (state_dict, num_samples)
        输出：
            聚合后的 state_dict
        """
        total_samples = sum(num_samples for _, num_samples in state_dicts_with_samples_list)

        # 初始化聚合模型参数（使用第一个 state_dict 的深拷贝）
        agg_state_dict = deepcopy(state_dicts_with_samples_list[0][0])
        for key in agg_state_dict:
            agg_state_dict[key] = agg_state_dict[key] * 0.0  # 清零

        # 加权累加
        for state_dict, num_samples in state_dicts_with_samples_list:
            weight = num_samples / total_samples
            for key in agg_state_dict:
                agg_state_dict[key] += state_dict[key] * weight

        return agg_state_dict

    def change_user_data(self, index):
        del self.train_data_loader
        del self.add_noise_loader
        self.train_data = self.train_data_list[index]

        # dataloader的构造部分
        self.train_data_loader = DataLoader(self.train_data, batch_size=self.batchsize, shuffle=True, num_workers=4)
        self.add_noise_loader = self.select_one_sample_per_class(self.train_data)
        self.data_num = len(self.train_data)

    def fit_normal(
            self, parameters, config
    ):

        self.model.load_state_dict(parameters)
        self.model.to(self.device)
        self.model.train()
        for epoch in range(config['epoch']):  # 简单训练 1 个 epoch
            for data, target in self.train_data_loader:
                # 清零梯度
                self.model.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)

                # 前向传播
                output = self.model(data)
                # 计算损失
                loss = self.criterion(output, target)
                # 反向传播
                loss.backward()
                # 手动更新权重（梯度更新）
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # 使用学习率控制步长，手动更新权重
                            param -= config.get('lr', 0.01) * param.grad
        self.model.to('cpu')
        return self.model.state_dict(), self.data_num, {"some_metric": 0.0}

    def fit_dp(self, parameters, config):
        """
        Local Differential Privacy (Client-level LDP) 客户端本地更新
        严格遵循经典伪代码流程：
        - 每个 batch：计算平均梯度 → clip → 更新参数
        - 所有 epoch 结束后：对最终参数 w_t+1 加高斯噪声 N(0, σ²I)
        - 上传带噪参数给服务器
        """
        # -----------------------------
        # 配置（与你提供的完全一致）
        # -----------------------------
        max_grad_norm = config.get("clip", 5.0)  # C: gradient clipping threshold
        epsilon_budget = config.get("epsilon", 8.0)  # 当前轮次或总预算（仅作参考）
        epochs = config.get("epoch", 10)  # local epochs E
        lr = config.get("lr", 0.01)  # learning rate η

        # -----------------------------
        # 1. 加载全局模型参数
        # -----------------------------
        self.model.load_state_dict(parameters)
        self.model.to(self.device)
        self.model.train()

        # -----------------------------
        # 2. 优化器
        # -----------------------------
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # -----------------------------
        # 3. 本地训练：多 epoch + per-batch clipping
        # -----------------------------
        for epoch in range(epochs):
            for data, target in self.train_data_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # ===== per-batch gradient clipping =====
                # 使用 PyTorch 官方工具，原地剪裁 batch 平均梯度到 L2 <= max_grad_norm
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=max_grad_norm,
                    norm_type=2.0
                )

                # 更新参数
                optimizer.step()

        # -----------------------------
        # 4. 计算噪声标准差并对最终参数加噪
        # -----------------------------

        batch = math.ceil(len(self.train_data_loader.dataset) / self.batchsize)
        # 敏感度：一次batch更新梯度为max_clip，n次更新就是n * max_clip
        dp_sigma = epochs * batch * lr * (max_grad_norm / self.batchsize) * math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon_budget
        print(dp_sigma) # 0.0048448052626053895 ——std



        noisy_state_dict = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # 每个参数独立加噪声 N(0, σ²)
                gaussian = torch.distributions.Normal(loc=0.0, scale=dp_sigma)  # scale=std
                noise = gaussian.sample(param.shape).to(
                    device=self.device,
                    dtype=param.dtype
                )
                noisy_state_dict[name] = param.data + noise

            # 处理 buffers（如 BN 的 running_mean/running_var）
            for name, buffer in self.model.named_buffers():
                if name not in noisy_state_dict:
                    noisy_state_dict[name] = buffer.clone().detach()

        # -----------------------------
        # 5. 返回带噪参数 + 额外信息
        # -----------------------------
        return noisy_state_dict, self.data_num, {
            "local_dp": True,
            "clip_norm": max_grad_norm,
            "local_epochs": epochs,
            "estimated_single_round_epsilon": None  # 可选：后续可添加估算函数
        }

    def fit_tip(self, parameters, config
                    ):
        # 加载全局模型参数
        # 安装论文里面的描述，先tmd正常训练，此过程中进行梯度剪裁，训练完再直接往参数加噪
        # 我tmd也不知道论文里面的算法为啥这么写
        print('一个客户端完成训练')
        max_clip = config.get('clip', 5.0)
        lr = config.get('lr', 0.01)
        epsilon = config.get('epsilon', 1)
        delta = config.get('delta', 1e-5)
        # 这里N代表次客户端使用的数据量，通过客户端对象的本地数据获取
        epoch = config.get('epoch', 1)

        self.model.load_state_dict(parameters)
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for e in range(epoch):  # 简单训练 1 个 epoch
            for data, target in self.train_data_loader:

                data = data.to(self.device)
                target = target.to(self.device)
                # 前向传播
                output = self.model(data)
                # 计算损失
                loss = self.criterion(output, target)
                loss.backward()
                # 遍历参数并更新（裁剪  + 手动更新）
                total_norm = torch.sqrt(
                    sum(p.grad.data.norm() ** 2 for p in self.model.parameters() if p.grad is not None))

                clip_coef = max_clip / (total_norm + 1e-6)
                clip_coef = min(1.0, clip_coef)

                if clip_coef < 1:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(clip_coef)

                optimizer.step()

        # 开始加噪
        # batch = math.ceil(500 / self.batchsize)
        # sigma = epoch * batch * lr * (max_clip / self.batchsize) * math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon
        # 使用差分隐私加噪方法：
        # 之前使用本地数据进行训练,经过了eopch轮次,使用batch,学习率为lr,进行了梯度剪裁

        # 这里假设有n个样本要加入噪音，每个样本对应噪音的强度因该是sigma / n

        batch = math.ceil(len(self.train_data_loader.dataset) / self.batchsize)
        beta = epoch * batch * lr * (max_clip / self.batchsize) * math.sqrt(
            2 * math.log(1.25 / 1e-5)) / epsilon
        beta = beta / len(self.add_noise_loader)
        print(beta,'asdf')
        # cam注册一次继续，因为内部实现是覆盖的
        cam = HookGradCam(self.model)
        for image, label in self.add_noise_loader:
            cam.register_hooks()
            # print(f'{self.device}-',config['fit_method'],'加噪音')
            self.model.zero_grad()
            image = image.to(self.device)
            label = label.to(self.device)
            # 注册钩子
            output = self.model(image)
            output[0, label].backward()
            cam.calculate_channel_weights()
            cam.make_noise_to_weight(beta=beta)

            # 开始手动加噪
            with torch.no_grad():
                for name in cam.hook_layer_names:
                    layer = self.model.get_submodule(name)
                    # 这里模拟往参数的频率域加入扰动

                    weight = layer.weight.data  # [C_out, C_in, H, W]
                    fft_weight = torch.fft.fft2(weight)
                    fft_shifted = torch.fft.fftshift(fft_weight, dim=(-2, -1))
                    fft_noised = fft_shifted + cam.weight_noise[name].to(weight.device)
                    # 回到空间域
                    fft_unshifted = torch.fft.ifftshift(fft_noised, dim=(-2, -1))
                    weight_noised = torch.fft.ifft2(fft_unshifted).real
                    layer.weight.data.copy_(weight_noised)

            cam.clear_data()
            cam.clear_hooks()
        # 加噪完成，给服务器传回去
        self.model.to('cpu')
        return self.model.state_dict(), self.data_num, {"some_metric": 0.0}

    def fit_apg(self, parameters, config
                ):
        # 加载全局模型参数
        # 安装论文里面的描述，先tmd正常训练，此过程中进行梯度剪裁，训练完再直接往参数加噪
        # 我tmd也不知道论文里面的算法为啥这么写
        max_clip = config.get('clip', 5.0)
        lr = config.get('lr', 0.01)
        epsilon = config.get('epsilon', 1)
        delta = config.get('delta', 1e-5)
        # 这里N代表次客户端使用的数据量，通过客户端对象的本地数据获取
        epoch = config.get('epoch', 1)

        self.model.load_state_dict(parameters)
        self.model.to(self.device)
        self.model.train()
        for epoch in range(epoch):  # 简单训练 1 个 epoch
            for data, target in self.train_data_loader:

                data = data.to(self.device)
                target = target.to(self.device)
                # 前向传播
                output = self.model(data)
                # 计算损失
                loss = self.criterion(output, target)
                grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False, retain_graph=False)
                # 遍历参数并更新（裁剪  + 手动更新）
                total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads if g is not None))
                clip_coef = max_clip / (total_norm + 1e-6)
                clip_coef = min(1.0, clip_coef)
                clipped_grads = tuple(g * clip_coef for g in grads)
                with torch.no_grad():
                    for (name, param), grad in zip(self.model.named_parameters(), clipped_grads):
                        # 更新
                        param -= lr * grad

        # 开始加噪

        # 这里假设有100个样本要加入噪音，每个样本对应噪音的强度因该是sigma / 100

        batch = math.ceil(len(self.train_data_loader.dataset) / self.batchsize)
        # 敏感度：一次batch更新梯度为max_clip，n次更新就是n * max_clip
        agp_sigma = epoch * batch * lr * (max_clip / self.batchsize) * math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon
        # cam注册一次继续，因为内部实现是覆盖的
        cam = HookGradCam(self.model)
        t = cam.hook_layer_names
        # 这里复现论文中目标层为每个残差块中最后一个卷积层
        c = []
        for i in t:
            if 'conv2' in i:
                c.append(i)
        cam.obg_layer = c



        for image, label in self.add_noise_loader:
            # 注册钩子
            cam.register_hooks()
            print(f'{self.device}-', config['fit_method'], '加噪音')
            self.model.zero_grad()
            image = image.to(self.device)
            label = label.to(self.device)
            output = self.model(image)
            output[0, label].backward()
            cam.calculate_channel_weights()
            cam.make_agp_noise_to_weight(beta=agp_sigma)

            # 开始手动加噪
            for name in cam.hook_layer_names:
                layer = self.model.get_submodule(name)
                layer.weight.data += cam.weight_AGP_noise[name] * lr

            cam.clear_data()
            cam.clear_hooks()
        # 加噪完成，给服务器传回去
        self.model.to('cpu')
        return self.model.state_dict(), self.data_num, {"some_metric": 0.0}

    def fit(self, parameters, config):
        #
        res_para = []

        rng = random.Random(int(time.time()))
        # 这个是相当于模拟选择用户（选择用100分数据集中的哪10份）
        rand_list = rng.sample(range(self.num_train_data_list), self.one_true_user_simulation_num)
        for i in range(self.one_true_user_simulation_num):
            # 先执行切换用户
            self.change_user_data(self.simpler.sample())
            if config['fit_method'] == 'normal':
                temp, data_num, _ = self.fit_normal(parameters, config)
            elif config['fit_method'] == 'dp':
                temp, data_num, _ = self.fit_dp(parameters, config)
            elif config['fit_method'] == 'agp':
                temp, data_num, _ = self.fit_apg(parameters, config)
            elif config['fit_method'] == 'tip':
                temp, data_num, _ = self.fit_tip(parameters, config)
            else:
                raise KeyError(r'''config['fit_method'] 的值不在预设之中''')
            # 这里的temp是模型的权重字典
            res_para.append((deepcopy(temp), data_num))

        avg_state_dict = self.aggregate(res_para)
        self.model.load_state_dict(avg_state_dict)
        return avg_state_dict, self.one_true_user_simulation_num * self.data_num, {'sdf': 23}

    def evaluate(self, parameters, config):
        # 这里评估聚合模型的的
        return 23, self.data_num, {"sdaf": 0}
