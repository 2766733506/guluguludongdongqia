import math

import torch.nn as nn
import random
import numpy as np
from torch.nn.grad import conv2d_weight
import torch.nn.functional as F
import torch
def add_high_freq_noise_to_grad(grad, beta=0.1, ratio=0.5):
    """
    grad: torch.Tensor, shape [H, W]，单通道二维梯度
    beta: 扰动强度系数
    ratio: 高频掩码半径比例，越大只保留更外围的高频
    """
    H, W = grad.shape
    device = grad.device

    # 1. 原始梯度做FFT
    freq = torch.fft.fft2(grad)

    # 2. 构造高频掩码
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    center_y, center_x = H // 2, W // 2
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = dist.max()
    mask = (dist >= ratio * max_dist).float()

    # 3. 生成随机复数噪声（高频）
    real = torch.randn(H, W, device=device)
    imag = torch.randn(H, W, device=device)
    noise = torch.complex(real, imag) * mask

    # 4. 高频噪声叠加到频谱
    freq_noisy = freq + noise * beta

    # 5. IFFT回到空间域，取实部
    grad_noisy = torch.fft.ifft2(freq_noisy).real

    return grad_noisy


def add_high_freq_noise_to_weight(shape, beta=0.01, ratio=0.5):
    """
    生成高频扰动：傅里叶变换 -> 保留高频区域 -> 逆变换
    输入: shape = [C_in, H, W]（每个通道的参数维度）
    """
    import torch.fft

    noise = torch.randn(shape)  # 空间域白噪声
    fft_noise = torch.fft.fft2(noise)  # 傅里叶变换到频域
    fft_shifted = torch.fft.fftshift(fft_noise)  # 中心化频谱

    _, H, W = shape
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    radius = ratio * min(H, W) / 2
    dist = ((X - center_x) ** 2 + (Y - center_y) ** 2).sqrt()
    mask = (dist >= radius).float()  # 高频掩码（浮点型）

    # 应用掩码，保留高频部分，保留复数形式
    high_freq_noise = fft_shifted * mask # 有广播机制

    return high_freq_noise * beta  # 返回频域的复数噪声（带缩放）



def mix_grad(model, grad_dict, lr=0.01):
    # model = deepcopy(models)
    # model.to('cup')
    with torch.no_grad():
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                layer = model.get_submodule(name)
                if isinstance(model.get_submodule(name), nn.Conv2d):
                    if layer.weight.grad is None:
                        continue  # 跳过没有梯度的参数

                    # 判断是否是卷积层的权重，并在字典中有替代梯度
                    if isinstance(model.get_submodule(name), nn.Conv2d) and name in grad_dict:
                        # 用自定义梯度替换
                        custom_grad = grad_dict[name].to(layer.weight.device)
                        layer.weight.data -= lr * custom_grad
                    else:
                        # 使用反向传播后的默认梯度
                        layer.weight.data -= lr * layer.weight.grad


def clip_feature_grad(feature_grad, clip_norm):
    # 按样本进行feature grad剪裁
    B = feature_grad.shape[0]
    grad_flat = feature_grad.view(B, -1)
    grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)  # [B,1]

    # 计算缩放比例，不超过1
    clip_coef = (clip_norm / (grad_norm + 1e-6)).clamp(max=1.0)

    # 只有范数超过clip_norm才会缩放，没超过保持原样
    clipped_grad = grad_flat * clip_coef

    clipped_grad = clipped_grad.view_as(feature_grad)
    return clipped_grad



class HookGradCam:

    def __init__(self, model, seed=None, clip_norm=5):
        """
        初始化存储，对于独立的一个batch，将所有的训练过程中的数据存在列表中，针对于每一个卷积层，
        包括每一个卷积的输入特征图，输出特征图，fature的梯度,
        传进来的模型会进行展平，无法保留残差链接
        :param model:
        :param seed:
        """
        self.no_obj_laver = None
        self.model = model
        self.seed = seed
        self.forward_hooks = []
        self.hook_layer_names = []
        self.obj_layer = []

        self.feature_maps_in = {}  # key: 层名，value: 特征图
        self.feature_maps_out = {}  # key: 层名，value: 特征图
        self.feature_grads = {}  # key: 层名，value: 梯度
        self.weight_grad = {}  # key：层名字，value: weight_grad

        self.channel_weights = {}  # key：层名，value：通道的权重（grad-cam）
        self.noise_feature_grad = {}  # key:层名，value:各个通道的noise，不加noise代表不加噪音
        self.hot_map = {}
        self.clip_norm  = clip_norm
        self.weight_noise ={} # 用于记录权重的高频噪音
        self.weight_AGP_noise = {} # 记录agp方法的noise

        if self.seed:
            # 保证可复现
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)




    # 注册钩子，拿到训练信息
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                if isinstance(module, nn.Conv2d):
                    self.hook_layer_names.append(name)

                    def make_forward_hook(layer_name):
                        def forward_hook(module, input, output):
                            self.feature_maps_out[layer_name] = output
                            self.feature_maps_in[layer_name] = input[0].cpu().detach()

                            def grad_hook(grad):
                                self.feature_grads[layer_name] = grad.detach()

                                return grad  # 改动梯度

                            output.register_hook(grad_hook)

                        return forward_hook

                    handle = module.register_forward_hook(make_forward_hook(name))
                    self.forward_hooks.append(handle)

    # 清除钩子
    def clear_hooks(self):
        for h in self.forward_hooks:
            h.remove()

    # 清除数据
    def clear_data(self):
        self.feature_maps_in.clear()
        self.feature_maps_out.clear()
        self.feature_grads.clear()
        self.weight_grad.clear()
        self.hot_map.clear()
        self.channel_weights.clear()
        self.noise_feature_grad.clear()



    def calculate_channel_weights(self,no_obj_laver_num=0):
        """
        计算每一层中每个通道的权重（基于 Grad-CAM 的思路）
        :return: 字典 {层名: Tensor[B, C]}，每个 batch 对应每个通道的权重
        """

        for key in self.hook_layer_names:
            grads = self.feature_grads[key]  # [B, C, H, W]
            if grads is None:
                continue
            weights = grads.mean(dim=(2, 3))  # [B, C]，对 H, W 做均值
            self.channel_weights[key] = weights  # 每个通道一个权重
        self.no_obj_laver = self.hook_layer_names[:no_obj_laver_num]

    def make_noise_to_grad(self, proportion=0.1, noise_fn=None, beta=0.01,ratio=0.5):
        """
           生成每一层对应的扰动张量字典，未被选择的通道填充为 0。
            ratio:高频噪声的半径
           返回：
               dict: {层名: Tensor[B, C, H, W]}，表示扰动（未选通道为0）
           """
        num = max(1,len(self.hook_layer_names))
        beta = beta / num
        for key in self.hook_layer_names:
            data = self.feature_grads[key]
            if data is None:
                continue
            B, C, H, W = data.shape
            device = data.device
            perturbation = torch.zeros_like(data)
            k = max(1, int(C * proportion))
            for b in range(B):
                if key not in self.no_obj_laver:
                    topk_indices = torch.topk(self.channel_weights[key][b], k=k, largest=True).indices
                    for c in topk_indices:
                        noise = add_high_freq_noise_to_grad(data[b,c],beta,ratio) *beta
                        perturbation[b, c] = noise
                else:
                    for c in range(len(self.channel_weights[key][b])):
                        noise = add_high_freq_noise_to_grad(data[b, c], beta, ratio) * beta
                        perturbation[b, c] = noise

            self.noise_feature_grad[key] = perturbation




    def add_noise_to_feature_grads(self):
        """
        :return:
        """
        for key in self.hook_layer_names:
            self.feature_grads[key] += self.noise_feature_grad[key]

    def make_hot_map(self, use_relu=True, size=(32, 32)):
        """
           生成 Grad-CAM 风格的热力图（saliency map），每一个卷积层都有
           :param feature_maps: Tensor[B, C, H, W]，卷积层输出
           :param channel_weights: Tensor[B, C]，每个通道的权重
           :param use_relu: 是否在最后使用 ReLU（默认是）
           :param size: 若指定，则使用双线性插值放大到该大小(用于放缩导输入图片的大小进行重叠) ->(height, width)
           :return: Tensor[B, H, W]，每个样本一张和力图
           """
        for key in self.hook_layer_names:
            B, C, H, W = self.feature_maps_out[key].shape
            weights = self.channel_weights[key].view(B, C, 1, 1)  # [B, C, 1, 1]，广播到特征图维度
            weighted_maps = self.feature_maps_out[key] * weights  # 加权
            saliency = weighted_maps.sum(dim=1)  # [B, H, W]，对通道求和
            if use_relu:
                saliency = F.relu(saliency)
            if size is not None:
                saliency = F.interpolate(saliency.unsqueeze(1), size=size, mode='bilinear', align_corners=False)
                saliency = saliency.squeeze(1)  # 去掉中间通道维度


            self.hot_map[key] = saliency

    def show_heatmap(self, images, key, save_name=None,size=(32,32)):
        """
        将热力图与原图叠加显示，支持多张图片批量处理
        :param size:
        :param save_name: 保存图片的名字（可选），如果是多张图片，自动按序号保存，如 a_0.png, a_1.png
        :param key: 使用哪一层的hot_map进行生成热力图
        :param images: 原图列表或Tensor，形状可为 [B, 3, H, W] 或 List[PIL]
        """
        import matplotlib.pyplot as plt
        saliency_map = self.hot_map[key][0]
        print(saliency_map.shape)#  [H, W]
        img = images[0]
        img_np = img.permute(1, 2, 0).cpu().detach().numpy()  # [H, W, C]
        plt.imshow(img_np)  # 原图使用 RGB 显示
        saliency = saliency_map
        saliency_np = saliency.cpu().detach().numpy()
        saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
        fig, ax = plt.subplots(figsize=(size[0]/100, size[0]/100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.imshow(img_np)
        plt.imshow(saliency_np, cmap='jet', alpha=0.5)
        plt.axis('off')
        # plt.gca().set_position([0, 0, 1, 1])
        if save_name:
            plt.savefig(save_name, bbox_inches='tight',pad_inches=0)
            plt.close()




    def make_noise_to_weight(self, proportion=0.1, beta=0.01, ratio=0.5):
        """
        生成每一层权重维度的扰动张量，未被选择的通道填充为 0。
        ratio: 高频噪声保留半径
        返回：
            dict: {层名: Tensor[C_out, C_in, H, W]}，表示扰动（未选通道为0）
        """
        num = max(1, len(self.hook_layer_names))
        beta = beta / num
        print(beta)
        for key in self.hook_layer_names:
            layer = self.model.get_submodule(key)
            if not hasattr(layer, "weight") or layer.weight is None:
                continue

            weight_shape = layer.weight.shape  # [C_out, C_in, kH, kW]
            C_out = weight_shape[0]
            perturbation = torch.zeros_like(layer.weight,dtype=torch.complex64)

            k = max(1, int(C_out * proportion))
            device = layer.weight.device
            for c in range(C_out):
                if key not in self.no_obj_laver:
                    if c in torch.topk(self.channel_weights[key].mean(dim=0), k=k, largest=True).indices:
                        noise = add_high_freq_noise_to_weight(weight_shape[1:], beta=beta, ratio=ratio).to(device)
                        perturbation[c] = noise
                else:
                    # 全通道加噪
                    # k = max(1, int(C_out * 1))
                    # if c in torch.topk(self.channel_weights[key].mean(dim=0), k=k, largest=True).indices:
                    #     noise = add_high_freq_noise_to_weight(weight_shape[1:], beta=beta, ratio=ratio).to(device)
                    #     perturbation[c] = noise
                        pass
            self.weight_noise[key] = perturbation

    # def make_agp_noise_to_weight(self, proportion=0.25, beta=0.00001):
    #     """
    #     生成每一层权重维度的扰动张量，未被选择的通道填充为 0。
    #     参数：
    #         proportion: 参与添加噪声的通道比例
    #         beta: 控制随机噪声幅度（乘在 torch.randn 上）(总幅度)
    #
    #     返回：
    #         self.weight_AGP_noise
    #     """
    #     # beta = beta / num  # 控制整体扰动强度
    #
    #     for key in self.hook_layer_names:
    #         layer = self.model.get_submodule(key)
    #         if not hasattr(layer, "weight") or layer.weight is None:
    #             continue
    #
    #         weight_shape = layer.weight.shape  # [C_out, C_in, kH, kW]
    #         C_out = weight_shape[0]
    #         device = layer.weight.device
    #
    #         perturbation = torch.zeros_like(layer.weight)
    #
    #         # 选择比例最小的通道
    #         k = max(1, int(C_out * proportion))
    #         low_importance_indices = torch.topk(self.channel_weights[key].mean(dim=0), k=k, largest=False).indices
    #
    #         for c in range(C_out):
    #             if c in low_importance_indices:
    #                 # 替换为随机噪声，而非高频噪声
    #                 pass
    #                 # laplace = torch.distributions.Laplace(loc=0.0, scale=noise_scale)
    #                 # noise = laplace.sample(self.channel_weights[key].shape).to(
    #                 #     device=self.device,
    #                 #     dtype=param.dtype
    #                 # perturbation[c] = noise
    #
    #         self.weight_AGP_noise[key] = perturbation

    import math
    import torch

    def make_agp_noise_to_weight(self, proportion=0.25, beta=1e-5, scale_by_batch=True):
        """
        一次性加噪版本：
        1) perturbation 只负责记录哪些输出通道需要加噪（mask）
        2) 统计完后，对这些通道只采样一次噪声并写入 perturbation
        """
        self.weight_AGP_noise = {}
        proportion = 1- proportion
        for key in self.hook_layer_names:
            layer = self.model.get_submodule(key)

            if not hasattr(layer, "weight") or layer.weight is None:
                print(f"skip layer {layer}")
                continue

            W = layer.weight
            C_out = W.shape[0]
            device = W.device
            dtype = W.dtype

            k = max(1, int(C_out * proportion))

            cw = self.channel_weights.get(key, None)  # [B, C_out]

            assert cw.shape[1] == C_out, f"[{key}] cw.shape[1]={cw.shape[1]} != C_out={C_out}"

            # --- 1) 统计 mask：哪些输出通道要加噪 ---
            # union_mask[c] = True 表示该通道至少在某个样本的 topk 中出现过
            union_mask = torch.zeros(C_out, dtype=torch.bool, device=device)

            if key  in self.obj_layer:
                # 每个样本取 top-k，做 union
                topk_idx = torch.topk(cw, k=k, dim=1, largest=False).indices  # [B, k]
                union_mask[topk_idx.reshape(-1)] = True
            else:
                # no_obj_layer：全部输出通道加噪
                union_mask[:] = True

            # --- 2) 一次性生成噪声，只对 union_mask 为 True 的通道加 ---
            perturbation = torch.zeros_like(W)

            # 你原来除以 sqrt(B) 的思路：
            # 如果希望 batch 越大噪声越弱（更平滑），就保留这个缩放
            # 如果希望噪声强度与 batch 无关，就不缩放
            # noise_std = beta / math.sqrt(B) if scale_by_batch else beta

            # 对选中的通道一次性加噪
            selected = union_mask.nonzero(as_tuple=True)[0]
            for c in selected.tolist():
                noise = torch.normal(mean=0.0, std=beta, size=W[c].shape,
                                     device=device, dtype=dtype)
                perturbation[c] = noise  # 注意：只加一次，所以直接赋值即可

            self.weight_AGP_noise[key] = perturbation

