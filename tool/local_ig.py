
import os
from collections import OrderedDict

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import recovery as rs
from recovery.nn import MetaMonkey

def new_plot(tensor, title="", path=None,image_shape=(3,32,32)):
    tensor = tensor.cpu()
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]

    if batch_size == 1:
        img = tensor[0]
        plt.figure(figsize=(2, 2))

        fig, ax = plt.subplots(figsize=(image_shape[1] / 100, image_shape[2] / 100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        if channels == 1:
            plt.imshow(img.squeeze(0), cmap='gray')  # 灰度图
        elif channels == 3:
            plt.imshow(img.permute(1, 2, 0))  # RGB 图
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
        if title:
            # plt.title(title)
            pass
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        fig, axes = plt.subplots(1, batch_size, figsize=(2 * batch_size, 2))
        for i, img in enumerate(tensor):
            ax = axes[i]
            ax.axis('off')
            if channels == 1:
                ax.imshow(img.squeeze(0), cmap='gray')
            elif channels == 3:
                ax.imshow(img.permute(1, 2, 0))
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()


def process_recons_results(result, ground_truth, figpath, recons_path, filename, ds=None,image_shape=(3,32,32)):
    output_list, stats, history_list, x_optimal = result
    x_optimal = x_optimal.detach().cpu()
    test_mse = (x_optimal - ground_truth.cpu()).pow(2).mean()
    # try:
    #     test_psnr = rs.metrics.psnr(x_optimal, ground_truth, factor=1 / ds, batched=True)
    # except RuntimeError:
    #     pass
    test_psnr = rs.metrics.psnr(x_optimal, ground_truth.cpu(), factor=1 / ds, batched=False)
    title = f"MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | "
    image = transforms.ToPILImage()(x_optimal[0,:,:])

    # 保存图片
    # image.save(f"{filename}.png")
    x_optimal = x_optimal.detach().cpu()  # ✅ 确保在 CPU
    ground_truth = ground_truth.detach().cpu()
    # new_plot(torch.cat([ground_truth, x_optimal[:,:,:]]), title, path=os.path.join(figpath, f'{filename}.png'))
    new_plot(x_optimal[:,:,:], title, path=os.path.join(figpath, f'{filename}_MSE-{test_mse:2.4f}_PSNR-{test_psnr:4.2f}.png'),image_shape=image_shape)
    torch.save({'output_list': output_list.cpu(), 'stats': stats, 'history_list': history_list, 'x_optimal': x_optimal},
               open(os.path.join(recons_path, f'{filename}.pth'), 'wb'))




def run_ig_attack(
    model_before, model_after, image_true, label_true,
    device='cuda:0',
    filename='ig_recover',
    img_shape=(3, 32, 32),
    figure_folder='.', recons_folder='rec', num_images=1, recons_config=None,
    class_name='cifar10'
):
    # 标准化参数（CIFAR-10）
    if recons_config is None:
        recons_config = dict(
            signed=True, boxed=True, cost_fn='sim',
            indices='def', weights='equal',
            lr=0.04, optim='adamw',
            restarts=1, max_iterations=15000,
            total_variation=1e-1, init='randn',
            filter='none', lr_decay=True,
            scoring_choice='pixelmean'
        )

    # cifar10的
    if class_name == 'cifar10':
        dm = torch.as_tensor([0.4914, 0.4822, 0.4465])[:, None, None]
        ds = torch.as_tensor([0.2023, 0.1994, 0.2010])[:, None, None]
    elif class_name == 'cifar100':
        dm = torch.as_tensor([0.5071598291397095, 0.4866936206817627, 0.44120192527770996])[:, None, None]
        ds = torch.as_tensor([0.2673342823982239, 0.2564384639263153, 0.2761504650115967])[:, None, None]
    elif class_name == 'bloodmnist':
        dm = torch.as_tensor([0.7944298982620239, 0.6597476601600647, 0.696285605430603])[:, None, None]
        ds = torch.as_tensor([0.2108030915260315, 0.2368169128894806, 0.11085140705108643])[:, None, None]
    elif class_name == 'tiny-imagenet-200':
        dm = torch.as_tensor([0.48023695374235836, 0.44806704707218714, 0.39750365085567174])[:, None, None]
        ds = torch.as_tensor([0.276436428750936, 0.268863279658034, 0.28158992889367984])[:, None, None]


    rec_machine = rs.GradientReconstructor(model_before, (dm, ds), recons_config, num_images=num_images)

    # 差值近似 ∇L ≈ θ_before - θ_after
    m0 = MetaMonkey(model_before)
    m1 = MetaMonkey(model_after)
    diff = OrderedDict(
        (name, param1 - param0)
        for (name, param0), (_, param1) in zip(m0.nparameters.items(), m1.nparameters.items())
    )

    diff_list = [v.detach().to(device) for v in diff.values()]


    # 重建图像
    rec_machine.model.eval()
    result = rec_machine.reconstruct(
        diff_list,
        # normalizer(image_true.to(setup['device'])),
        (image_true.to(device)),
        label_true.to(device),
        img_shape=img_shape
    )

    # 保存图像和中间结果
    process_recons_results(
        result, image_true,
        figpath=figure_folder,
        recons_path=recons_folder,
        filename=filename,
        ds=ds,
        image_shape=img_shape
    )

