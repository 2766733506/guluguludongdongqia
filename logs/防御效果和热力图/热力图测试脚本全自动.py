import math
import shutil
from copy import deepcopy

from torch.nn import CrossEntropyLoss
from torchvision import transforms
from tool.img2tensor import images_to_batch
from tool.locla_cam import HookGradCam
import torch
from tool.test_acc import test_cifar10_accuracy, test_cifar100_accuracy, test_stl10_accuracy
import os

# cifar10 的数据预处理
dm = torch.as_tensor([0.4914, 0.4822, 0.4465])[:, None, None]
ds = torch.as_tensor([0.2023, 0.1994, 0.2010])[:, None, None]
normalizer = transforms.Normalize(dm.squeeze(), ds.squeeze())


# # MNIST 数据集是单通道灰度图像，其均值和标准差如下（基于训练集统计值）
# dm = torch.as_tensor([0.1307])[:, None, None]  # 均值
# ds = torch.as_tensor([0.3081])[:, None, None]  # 标准差
# 
# 创建 stl10 的 Normalizer
# dm = torch.as_tensor([0.44671064615249634, 0.4398098886013031, 0.4066464304924011])[:, None, None]
# ds = torch.as_tensor([0.26034098863601685, 0.2565772831439972, 0.2712673842906952])[:, None, None]
# normalizer = transforms.Normalize(dm.squeeze(), ds.squeeze())
def make_naive_train(image, label, un_train_model_path, class_name=r'cifar10',
                     device='cuda:0'):
    # if class_name == 'cifar10':
    #     tester = test_cifar10_accuracy
    # elif class_name == 'cifar100':
    #     tester = test_cifar100_accuracy
    # elif class_name == 'stl10':
    #     tester = test_stl10_accuracy
    # else:
    #     raise KeyError('class_name key error')

    a = torch.load(un_train_model_path, weights_only=False)
    # acc = tester(a, device=device)
    # print(f'原始模型的acc：{acc}')
    image = image.to(device)
    label = label.to(device)
    loss_fn = CrossEntropyLoss()
    a.eval()
    a.to(device)
    lr = 0.001
    optimizer = torch.optim.SGD(a.parameters(), lr=lr,momentum=0)
    for epoch in range(4):
        optimizer.zero_grad()

        output = a(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
    # acc = tester(a, device=device)
    # print(f'normal的acc：{acc}')
    return a


def make_dp_train(image, label, un_train_model_path, class_name=r'cifar10',
                  device='cuda:0', epsilon=1, max_clip=1):
    # if class_name == 'cifar10':
    #     tester = test_cifar10_accuracy
    # elif class_name == 'cifar100':
    #     tester = test_cifar100_accuracy
    # elif class_name == 'stl10':
    #     tester = test_stl10_accuracy
    # else:
    #     raise KeyError('class_name key error')

    a = torch.load(un_train_model_path, weights_only=False).to(device)
    image = image.to(device)
    label = label.to(device)
    loss_fn = CrossEntropyLoss()
    max_grad_norm = 5
    a.eval()
    lr = 0.001
    optimizer = torch.optim.SGD(a.parameters(), lr=lr)

    # -----------------------------
    # 3. 本地训练：多 epoch + per-batch clipping
    # -----------------------------
    for epoch in range(4):
        data, target = image.to(device), label.to( device)

        optimizer.zero_grad()
        output = a(data)
        loss = loss_fn(output, target)
        loss.backward()

        # ===== per-batch gradient clipping =====
        # 使用 PyTorch 官方工具，原地剪裁 batch 平均梯度到 L2 <= max_grad_norm
        torch.nn.utils.clip_grad_norm_(
            parameters=a.parameters(),
            max_norm=max_grad_norm,
            norm_type=2.0
        )

        # 更新参数
        optimizer.step()

    # -----------------------------
    # 4. 计算噪声标准差并对最终参数加噪
    # -----------------------------


    batch = 1
    # 敏感度：一次batch更新梯度为max_clip，n次更新就是n * max_clip
    dp_sigma = 4 * batch * lr * (max_grad_norm / 1) * math.sqrt(
        2 * math.log(1.25 / 1e-5)) / epsilon

    noisy_state_dict = {}
    with torch.no_grad():
        for name, param in a.named_parameters():
            # 每个参数独立加噪声 N(0, σ²)
            noise = torch.normal(mean=0.0, std=dp_sigma, size=param.shape,
                         device=device, dtype=param.dtype)
            noisy_state_dict[name] = param.data + noise

        # 处理 buffers（如 BN 的 running_mean/running_var）
        for name, buffer in a.named_buffers():
            if name not in noisy_state_dict:
                noisy_state_dict[name] = buffer.clone().detach()
    a.load_state_dict(noisy_state_dict)

    # acc = tester(a, device=device)
    # print(f'dp模型的acc：{acc}')
    return a

def make_agp_train(image, label, un_train_model_path, class_name=r'cifar10',
                  device='cuda:0', epsilon=1, max_clip=1):
    # if class_name == 'cifar10':
    #     tester = test_cifar10_accuracy
    # elif class_name == 'cifar100':
    #     tester = test_cifar100_accuracy
    # elif class_name == 'stl10':
    #     tester = test_stl10_accuracy
    # else:
    #     raise KeyError('class_name key error')

    a = torch.load(un_train_model_path, weights_only=False).to(device)
    a.eval()
    lr = 0.001
    optimizer = torch.optim.SGD(a.parameters(), lr=lr, momentum=0)
    # 先梯度剪裁训练
    loss_fn = CrossEntropyLoss()
    for epoch in range(4):
        optimizer.zero_grad()
        output = a(image)
        loss = loss_fn(output, label)
        loss.backward()  # 计算梯度，存在 .grad
        # 计算所有参数梯度范数
        total_norm = torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in a.parameters() if p.grad is not None))

        clip_coef = max_clip / (total_norm + 1e-6)
        clip_coef = min(1.0, clip_coef)
        if clip_coef < 1:
            for p in a.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        optimizer.step()

    # agp_sigma = epoch * batch * lr * (max_clip / self.batchsize) * math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon
    agp_sigma = 4 * 1 * lr * (max_clip / 1) * math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon
    # cam注册一次继续，因为内部实现是覆盖的
    cam = HookGradCam(a)
    t = cam.hook_layer_names
    # 这里复现论文中目标层为每个残差块中最后一个卷积层
    c = []
    for i in t:
        if 'conv2' in i:
            c.append(i)
    cam.obg_layer = c


    # 注册钩子
    cam.register_hooks()
    a.zero_grad()
    image = image.to(device)
    label = label.to(device)
    output = a(image)
    output[0, label].backward()
    cam.calculate_channel_weights()
    print(f'agp{agp_sigma}')
    cam.make_agp_noise_to_weight(beta=agp_sigma)

    # 开始手动加噪
    for name in cam.hook_layer_names:
        layer = a.get_submodule(name)
        layer.weight.data += cam.weight_AGP_noise[name]

    cam.clear_data()
    cam.clear_hooks()
    # 加噪完成，给服务器传回去
    # acc = tester(a, device=device)
    # print(f'agp的acc：{acc}')
    return a

def make_suyiln_train(image, label, un_train_model_path, class_name=r'cifar10',
                  device='cuda:0', max_clip=1,config=None,proportion=0.3):
    # if class_name == 'cifar10':
    #     tester = test_cifar10_accuracy
    # elif class_name == 'cifar100':
    #     tester = test_cifar100_accuracy
    # elif class_name == 'stl10':
    #     tester = test_stl10_accuracy
    # else:
    #     raise KeyError('class_name key error')

    a = torch.load(un_train_model_path, weights_only=False).to(device)
    # acc = tester(a, device=device)
    # print(f'原始模型的acc：{acc}')
    max_clip = max_clip
    lr = config.get('lr', 0.001)
    epsilon = config.get('epsilon', 1)
    delta = config.get('delta', 1e-5)
    # 这里N代表次客户端使用的数据量，通过客户端对象的本地数据获取

    a = torch.load(un_train_model_path, weights_only=False).to(device)
    a.eval()
    a.to(device)

    optimizer = torch.optim.SGD(a.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    for epoch in range(4):  # 简单训练 1 个 epoch

        data = image.to(device)
        target = label.to(device)
        # 前向传播
        output = a(data)
        # 计算损失
        loss = loss_fn(output, target)
        loss.backward()
        # 遍历参数并更新（裁剪  + 手动更新）
        total_norm = torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in a.parameters() if p.grad is not None))

        clip_coef = max_clip / (total_norm + 1e-6)
        clip_coef = min(1.0, clip_coef)

        if clip_coef < 1:
            for p in a.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        optimizer.step()

    # 开始加噪
    batch = 1
    batchsize = 1
    beta = 4 * batch * lr * (max_clip / batchsize) * math.sqrt(
        2 * math.log(1.25 / 1e-5)) / epsilon
    print(beta) # 0.096 约等于0.1
    noise_scale = beta
    # 使用差分隐私加噪方法：
    # 之前使用本地数据进行训练,经过了eopch轮次,使用batch,学习率为lr,进行了梯度剪裁

    # 这里假设有100个样本要加入噪音，每个样本对应噪音的强度因该是sigma / 100
    beta = noise_scale
    # cam注册一次继续，因为内部实现是覆盖的
    cam = HookGradCam(a)
    cam.register_hooks()
    a.zero_grad()
    output = a(image)
    output[0, label].backward()
    cam.calculate_channel_weights(no_obj_laver_num=0)
    cam.make_noise_to_weight(beta=beta,proportion=proportion)

    # 开始手动加噪
    with torch.no_grad():
        for name in cam.hook_layer_names:
            layer = a.get_submodule(name)
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
    # acc = tester(a, device=device)
    # print(f'suyilin的acc：{acc}')
    return a

# 实验图片名字
def make_log(image_path, un_train_model_path,
             base_path=r'/home/Suyilin/paper_code/grad_cam_plus/logs/防御效果和热力图/cifar10', class_name=r'cifar10',
             image_size=(32, 32),
             device='cuda:0', epsilon=5):
    # 要测试的图片数据
    # images_to_batch之后返回的就已经是tensor了
    image, label = images_to_batch([image_path])
    if class_name == 'cifar10':
        transform = transforms.Compose(
        [transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
                                                     (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))])
    elif class_name =='cifar100':
        transform = transforms.Compose([
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            )
        ])
    elif class_name =='bloodmnist':
        transform = transforms.Compose([
            transforms.Normalize(
                mean=(0.7944298982620239, 0.6597476601600647, 0.696285605430603),
                std=(0.2108030915260315, 0.2368169128894806, 0.11085140705108643),
            )
        ])
    elif class_name == 'tiny-imagenet-200':
        transform = transforms.Compose([
            transforms.Normalize(
                mean=(0.48023695374235836, 0.44806704707218714, 0.39750365085567174),
                std=(0.276436428750936, 0.268863279658034, 0.28158992889367984),
            )
        ])

    else:
        raise KeyError('数据集不对')
    origin_image = deepcopy(image)
    origin_image= origin_image.to(device)
    image =transform(image)
    image = image.to(device)
    label = label.to(device)
    # ===================
    # 处理路径，进行自动化命名
    # ===================
    image_name = os.path.basename(image_path)  # asdlfjk.png
    image_name = image_name.split('.')[0]

    # 制造针对此样本的文件夹
    os.makedirs(os.path.join(base_path, image_name), exist_ok=True)
    os.makedirs(os.path.join(base_path, image_name, 'rec'), exist_ok=True)
    image_result_save_path = os.path.join(base_path, image_name)
    shutil.copy(image_path, image_result_save_path)
    # normal train

    naive_model = make_naive_train(image, label, deepcopy(un_train_model_path),
                                   class_name,
                                   device)
    naive_model.eval()
    naive_model.zero_grad(set_to_none=True)
    cam_normal = HookGradCam(naive_model)
    # 注册钩子
    cam_normal.register_hooks()
    output = naive_model(image)
    target = output.argmax(dim=1).item()


    output[0, target].sum().backward()
    # 先计算一波通道权重
    cam_normal.calculate_channel_weights()
    # 生产一下归因图
    cam_normal.make_hot_map(size=image_size)
    # 展示一下归因图
    # 之后看朴素训练完的图像的热力图
    # 这里产生normal的热力图
    print(cam_normal.hook_layer_names[-1])
    cam_normal.show_heatmap(origin_image, cam_normal.hook_layer_names[-1],
                     os.path.join(image_result_save_path, '*_normal.png'.replace('*', image_name)),size=image_size)

    # 清除钩子, 保存下来这个极其朴素的模型,保存的是整个模型
    cam_normal.clear_hooks()
    torch.save(naive_model,
               os.path.join(image_result_save_path, 'resnet18_train_one_simple_normal_*.pth'.replace('*', image_name)))

    # dp

    dp_model = make_dp_train(image, label, deepcopy(un_train_model_path),
                             class_name,
                             device, epsilon, max_clip=5)
    dp_model.eval()
    dp_model.zero_grad(set_to_none=True)
    cam = HookGradCam(dp_model)
    # 注册钩子
    cam.register_hooks()
    output = dp_model((image))
    output[0, label].sum().backward()
    # 先计算一波通道权重
    cam.calculate_channel_weights()
    # 生产一下归因图
    cam.make_hot_map(size=image_size)
    # 展示一下归因图
    # 之后看朴素训练完的图像的热力图
    # 这里产生normal的热力图
    cam.show_heatmap(origin_image, cam.hook_layer_names[-1],
                     os.path.join(image_result_save_path, '*_dp.png'.replace('*', image_name)),size=image_size)
    # 清除钩子, 保存下来这个极其朴素的模型,保存的是整个模型
    cam.clear_hooks()
    torch.save(dp_model,
               os.path.join(image_result_save_path, 'resnet18_train_one_simple_dp_*.pth'.replace('*', image_name)))


    # TIP 要分噪重要通道比例：30，60，90
    for i in [0.1,0.3,0.6]:
        suyiln_model = make_suyiln_train(image, label, deepcopy(un_train_model_path),
                                       class_name,
                                       device,max_clip=5,config={},proportion=i)
        suyiln_model.eval()
        suyiln_model.zero_grad(set_to_none=True)
        suyiln_cam = HookGradCam(suyiln_model)
        # 注册钩子
        suyiln_cam.register_hooks()
        output = suyiln_model((image))
        target = output.argmax(dim=1).item()
        output[0, target].sum().backward()
        # 先计算一波通道权重
        suyiln_cam.calculate_channel_weights()
        # 生产一下归因图
        suyiln_cam.make_hot_map(size=image_size)
        # 展示一下归因图
        # 之后看朴素训练完的图像的热力图
        # 这里产生normal的热力图
        suyiln_cam.show_heatmap(origin_image, suyiln_cam.hook_layer_names[-1],
                         os.path.join(image_result_save_path, '*_suyilin_$.png'.replace('*', image_name).replace('$', str(i))),size=image_size)

        # 清除钩子, 保存下来这个极其朴素的模型,保存的是整个模型
        suyiln_cam.clear_hooks()
        torch.save(suyiln_model,
                   os.path.join(image_result_save_path, 'resnet18_train_one_simple_suyilin_$_*.pth'.replace('$', str(i)).replace('*', image_name)))


    # agp
    agp_model = make_agp_train(image, label, deepcopy(un_train_model_path),
                             class_name,
                             device, epsilon, max_clip=5)
    agp_model.eval()
    agp_model.zero_grad(set_to_none=True)
    cam_apg = HookGradCam(agp_model)
    # 注册钩子
    cam_apg.register_hooks()
    output = agp_model((image))
    target = output.argmax(dim=1).item()
    output[0, target].sum().backward()
    # 先计算一波通道权重
    cam_apg.calculate_channel_weights()
    # 生产一下归因图
    cam_apg.make_hot_map(size=image_size)
    # 展示一下归因图
    # 之后看朴素训练完的图像的热力图
    # 这里产生normal的热力图
    cam_apg.show_heatmap(origin_image, cam_apg.hook_layer_names[-1],
                     os.path.join(image_result_save_path, '*_apg.png'.replace('*', image_name)),size=image_size)
    # 清除钩子, 保存下来这个极其朴素的模型,保存的是整个模型
    cam_apg.clear_hooks()
    torch.save(agp_model,
               os.path.join(image_result_save_path, 'resnet18_train_one_simple_apg_*.pth'.replace('*', image_name)))

