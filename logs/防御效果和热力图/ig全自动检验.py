import os

import torch
from sympy import print_glsl

from GIFD_Gradient_Inversion_Attack.auto_gifd import run_gifd_attac
from tool.img2tensor import images_to_batch
from tool.local_ig import run_ig_attack


def auto_ig(image_path, un_train_model_path,
            base_path=r'/home/Suyilin/paper_code/grad_cam_plus/logs/防御效果和热力图/cifar10',
            class_name=r'cifar10',
            device='cuda:0', img_shape=(3, 32, 32)):
    image_name = os.path.basename(image_path)  # asdlfjk.png
    image_name = image_name.split('.')[0]
    # 制造针对此样本的文件夹
    os.makedirs(os.path.join(base_path, image_name), exist_ok=True)
    os.makedirs(os.path.join(base_path, image_name, 'rec'), exist_ok=True)
    image_result_save_path = os.path.join(base_path, image_name)

    image, label = images_to_batch([image_path])
    image = image.to(device)
    label = label.to(device)
    print(label)

    approx = torch.load(un_train_model_path,
                        weights_only=False).to(device)
    approx.eval()

    recons_config_ig = dict(
        signed=True, boxed=True,
        # cost_fn='simlocal',
        cost_fn='sim',
        # cost_fn='l2',
        indices='def', weights='equal',
        lr=0.004, optim='adam',
        restarts=1, max_iterations=10000,
        total_variation=1e-4, init='randn',
        filter='none', lr_decay=True,
        scoring_choice='loss'
    )
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_normal_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()

    print('\nnormal\n')
    # normal效果
    run_ig_attack(origin, approx, image, label, filename='normal_*_ig_recover'.replace('*', image_name),
                  num_images=1, recons_config=recons_config_ig,
                  figure_folder=image_result_save_path,
                  recons_folder=os.path.join(image_result_save_path, 'rec'), img_shape=img_shape,
                  device=device, class_name=class_name
                  )
    # suyilin效果
    print('\nsuyilin\n')
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_suyilin_0.1_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    run_ig_attack(origin, approx, image, label, filename='suyiilin_0.1_*_ig_recover'.replace('*', image_name),
                  num_images=1, recons_config=recons_config_ig,
                  figure_folder=image_result_save_path,
                  recons_folder=os.path.join(image_result_save_path, 'rec'),
                  img_shape=img_shape,
                  device=device, class_name=class_name
                  )
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_suyilin_0.3_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    run_ig_attack(origin, approx, image, label, filename='suyiilin_0.3_*_ig_recover'.replace('*', image_name),
                  num_images=1, recons_config=recons_config_ig,
                  figure_folder=image_result_save_path,
                  recons_folder=os.path.join(image_result_save_path, 'rec'),
                  img_shape=img_shape,
                  device=device, class_name=class_name
                  )

    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_suyilin_0.6_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    run_ig_attack(origin, approx, image, label, filename='suyiilin_0.6_*_ig_recover'.replace('*', image_name),
                  num_images=1, recons_config=recons_config_ig,
                  figure_folder=image_result_save_path,
                  recons_folder=os.path.join(image_result_save_path, 'rec'),
                  img_shape=img_shape,
                  device=device, class_name=class_name
                  )

    # apg效果
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_apg_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    print('\napg\n')
    run_ig_attack(origin, approx, image, label, filename='apg_*_ig_recover'.replace('*', image_name),
                  num_images=1, recons_config=recons_config_ig,
                  figure_folder=image_result_save_path,
                  recons_folder=os.path.join(image_result_save_path, 'rec'),
                  img_shape=img_shape,
                  device=device, class_name=class_name
                  )

    # dp效果
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_dp_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    print('\ndp\n')
    run_ig_attack(origin, approx, image, label, filename='dp_*_ig_recover'.replace('*', image_name),
                  num_images=1, recons_config=recons_config_ig,
                  figure_folder=image_result_save_path,
                  recons_folder=os.path.join(image_result_save_path, 'rec'),
                  img_shape=img_shape,
                  device=device, class_name=class_name
                  )

    # 不要dlg
    # dlg配置
    # recons_config_dlg = dict(
    #     signed=True, boxed=True,
    #     cost_fn='l2',
    #     indices='def', weights='equal',
    #     lr=0.004, optim='adam',
    #     restarts=1, max_iterations=10000,
    #     total_variation=1e-4, init='randn',
    #     filter='none', lr_decay=True,
    #     scoring_choice='loss'
    # )
    # # 这个是未学习目标样本的模型，逻辑上近似等于遗忘模型
    #
    # origin = torch.load(
    #     os.path.join(image_result_save_path, 'resnet18_train_one_simple_normal_*.pth'.replace('*', image_name)),
    #     weights_only=False).to(device)
    # origin.eval()
    #
    # print('\nnormal\n')
    # # normal效果
    # run_ig_attack(origin, approx, image, label, filename='normal_*_dlg_recover'.replace('*', image_name),
    #               num_images=1, recons_config=recons_config_dlg,
    #               figure_folder=image_result_save_path,
    #               recons_folder=os.path.join(image_result_save_path, 'rec'),
    #               device=device,class_name=class_name,img_shape=img_shape
    #               )
    # # suyilin效果
    # origin = torch.load(
    #     os.path.join(image_result_save_path, 'resnet18_train_one_simple_suyilin_0.3_*.pth'.replace('*', image_name)),
    #     weights_only=False).to(device)
    # origin.eval()
    # print('\nsuyilin\n')
    # run_ig_attack(origin, approx, image, label, filename='suyiilin_*_dlg_recover'.replace('*', image_name),
    #               num_images=1, recons_config=recons_config_dlg,
    #               figure_folder=image_result_save_path,
    #               recons_folder=os.path.join(image_result_save_path, 'rec'),
    #               img_shape=img_shape,
    #               device=device,class_name=class_name
    #               )
    #
    #
    # # apg效果
    # origin = torch.load(
    #     os.path.join(image_result_save_path, 'resnet18_train_one_simple_apg_*.pth'.replace('*', image_name)),
    #     weights_only=False).to(device)
    # origin.eval()
    # print('\napg\n')
    # run_ig_attack(origin, approx, image, label, filename='apg_*_dlg_recover'.replace('*', image_name),
    #               num_images=1, recons_config=recons_config_dlg,
    #               figure_folder=image_result_save_path,
    #               recons_folder=os.path.join(image_result_save_path, 'rec'),
    #               img_shape=img_shape,
    #               device=device,class_name=class_name
    #               )
    #
    # # dp效果
    # origin = torch.load(
    #     os.path.join(image_result_save_path, 'resnet18_train_one_simple_dp_*.pth'.replace('*', image_name)),
    #     weights_only=False).to(device)
    # origin.eval()
    # print('\ndp\n')
    # run_ig_attack(origin, approx, image, label, filename='dp_*_dlg_recover'.replace('*', image_name),
    #               num_images=1, recons_config=recons_config_dlg,
    #               figure_folder=image_result_save_path,
    #               recons_folder=os.path.join(image_result_save_path, 'rec'),
    #               img_shape=img_shape,
    #               device=device,class_name=class_name
    #               )


def auto_gifd(image_path, un_train_model_path,
              base_path=r'/home/Suyilin/paper_code/grad_cam_plus/logs/防御效果和热力图/cifar10',
              class_name=r'cifar10',
              device='cuda:0', img_shape=(3, 32, 32)):
    image_name = os.path.basename(image_path)  # asdlfjk.png
    image_name = image_name.split('.')[0]
    # 制造针对此样本的文件夹
    os.makedirs(os.path.join(base_path, image_name), exist_ok=True)
    os.makedirs(os.path.join(base_path, image_name, 'GIFD_normal'), exist_ok=True)
    os.makedirs(os.path.join(base_path, image_name, 'GIFD_dp'), exist_ok=True)
    os.makedirs(os.path.join(base_path, image_name, 'GIFD_apg'), exist_ok=True)
    image_result_save_path = os.path.join(base_path, image_name)

    image, label = images_to_batch([image_path])
    image = image.to(device)
    label = label.to(device)
    print(label)

    approx = torch.load(un_train_model_path,
                        weights_only=False).to(device)
    approx.eval()
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_normal_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()

    print('\nnormal\n')
    # normal效果
    run_gifd_attac(origin, approx, image, label, class_name, device=device,
                   res_save_path=os.path.join(base_path, image_name, 'GIFD_normal'))

    # # TIP效果
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_suyilin_0.1_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    run_gifd_attac(origin, approx, image, label, class_name, device=device,
                   res_save_path=os.path.join(base_path, image_name, 'GIFD_TIP_0.1'))

    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_suyilin_0.3_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    run_gifd_attac(origin, approx, image, label, class_name, device=device,
                   res_save_path=os.path.join(base_path, image_name, 'GIFD_TIP_0.3'))

    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_suyilin_0.6_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    run_gifd_attac(origin, approx, image, label, class_name, device=device,
                   res_save_path=os.path.join(base_path, image_name, 'GIFD_TIP_0.6'))

    # apg效果
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_apg_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    run_gifd_attac(origin, approx, image, label, class_name, device=device,
                   res_save_path=os.path.join(base_path, image_name, 'GIFD_apg'))

    # dp效果
    origin = torch.load(
        os.path.join(image_result_save_path, 'resnet18_train_one_simple_dp_*.pth'.replace('*', image_name)),
        weights_only=False).to(device)
    origin.eval()
    run_gifd_attac(origin, approx, image, label, class_name, device=device,
                   res_save_path=os.path.join(base_path, image_name, 'GIFD_dp'))
