from PIL import Image
import torchvision.transforms as transforms
import torch
import os

from torch.utils.data import Dataset


def images_to_batch(image_paths):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 [C, H, W]，像素归一化到 [0, 1]
        # 这里不进行任何的归一化
    ])

    images = []
    labels = []
    for path in image_paths:
        label = os.path.basename(path)
        print(label)
        label = label.split('_')[-1]
        label = label.replace('.png','')
        label = eval(label)
        labels.append(label)
        img = Image.open(path).convert('RGB')  # 确保是 RGB 三通道
        # img = Image.open(path).convert('L')
        img_tensor = transform(img)
        images.append(img_tensor)

    # 堆叠成一个 Batch，形状为 [B, C, H, W]
    batch = torch.stack(images)
    labels = torch.tensor(labels)
    return batch, labels

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图片
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # 应用预处理操作
        if self.transform:
            img = self.transform(img)

        # 获取标签
        label = self.labels[idx]
        return img, label
