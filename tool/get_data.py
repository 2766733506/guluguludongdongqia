import torch
from torchvision import datasets, transforms
def get_cifar10_class_samples(num_samples=4):
    assert 1 <= num_samples <= 10, "最多只能获取10个不同类的样本（CIFAR-10 有 10 个类别）"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    class_to_img = {}
    images = []
    labels = []

    for img, label in dataset:
        if label not in class_to_img:
            class_to_img[label] = img
            images.append(img)
            labels.append(label)

        if len(class_to_img) == num_samples:
            break

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    return images_tensor, labels_tensor


import torch
from torchvision import datasets, transforms
from PIL import Image
import os

def get_and_save_mnist_class_samples(output_dir="./mnist_samples"):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 Tensor，范围 0-1
    ])

    dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    class_to_img = {}
    for img_tensor, label in dataset:
        if label not in class_to_img:
            class_to_img[label] = img_tensor
        if len(class_to_img) == 10:
            break

    # 保存每个类别的一张图片
    for label, img_tensor in class_to_img.items():
        img_pil = transforms.ToPILImage()(img_tensor)
        save_path = os.path.join(output_dir, f"mnist_{label}.png")
        img_pil.save(save_path)
        print(f"Saved {save_path}")

get_and_save_mnist_class_samples()
