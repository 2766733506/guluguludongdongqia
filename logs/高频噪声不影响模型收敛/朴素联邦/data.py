import random

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from collections import defaultdict

def split_datasets_dirichlet(num=100, type='cifar10', alpha=0.5, seed=0, min_size=10):
    """
    使用Dirichlet分布将训练集划分为num个子数据集（非IID）。
    - num: 子集数量（比如联邦学习里的客户端数）
    - alpha: Dirichlet浓度参数，越小越非IID
    - seed: 随机种子
    - min_size: 每个子集至少要有多少样本，避免空集/过小（不满足会重采样）
    """
    rng = np.random.default_rng(seed)

    # Step 1: Load dataset
    if type == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
                (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)
            )
        ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform
        ])
        data = torchvision.datasets.CIFAR10(
            root='/home/Suyilin/paper_code/grad_cam_plus/data',
            train=True, download=True, transform=transform_train
        )
        num_classes = 10

    elif type == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592378616, 0.48654890060424805, 0.44091784954071045),
                (0.2673340141773224, 0.2564387023448944, 0.2761503756046295)
            )
        ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform
        ])
        data = torchvision.datasets.CIFAR100(
            root='/home/Suyilin/paper_code/grad_cam_plus/data',
            train=True, download=True, transform=transform_train
        )
        num_classes = 100
    else:
        raise ValueError("type must be 'cifar10' or 'cifar100'")

    # Step 2: Organize indices by class (注意：这里取label会触发transform，但label不变；如想更快可用data.targets)
    class_indices = defaultdict(list)
    if hasattr(data, "targets"):
        targets = data.targets
        for idx, label in enumerate(targets):
            class_indices[int(label)].append(idx)
    else:
        for idx in range(len(data)):
            _, label = data[idx]
            class_indices[int(label)].append(idx)

    # Step 3: Dirichlet split (repeat until each subset has at least min_size samples)
    while True:
        client_indices = [[] for _ in range(num)]

        for c in range(num_classes):
            idxs = np.array(class_indices[c])
            rng.shuffle(idxs)

            # 对该类样本，给num个子集采样比例
            proportions = rng.dirichlet(alpha * np.ones(num))

            # 将该类样本按比例切分
            split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            splits = np.split(idxs, split_points)

            for i in range(num):
                client_indices[i].extend(splits[i].tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            break
        # 不满足最小样本数，重采样（保持seed不变会一直一样，所以这里推进rng状态已足够）

    # Step 4: Create Subset list
    sub_datasets = []
    for i in range(num):
        sub_datasets.append(Subset(data, np.array(client_indices[i], dtype=int)))

    return sub_datasets

def split_datasets_class_balanced(num=100, type='cifar10', seed=0, min_size=10):
    """
    类别平衡划分训练集为 num 个子数据集（尽量保证每个子集的类别分布一致）。
    - num: 子集数量（客户端数）
    - seed: 随机种子
    - min_size: 每个子集至少样本数（不满足会报错，因为平衡切分本身是确定性的）
    """
    rng = np.random.default_rng(seed)

    # Step 1: Load dataset
    if type == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
                (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)
            )
        ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform
        ])
        data = torchvision.datasets.CIFAR10(
            root='/home/Suyilin/paper_code/grad_cam_plus/data',
            train=True, download=True, transform=transform_train
        )
        num_classes = 10

    elif type == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592378616, 0.48654890060424805, 0.44091784954071045),
                (0.2673340141773224, 0.2564387023448944, 0.2761503756046295)
            )
        ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform
        ])
        data = torchvision.datasets.CIFAR100(
            root='/home/Suyilin/paper_code/grad_cam_plus/data',
            train=True, download=True, transform=transform_train
        )
        num_classes = 100
    else:
        raise ValueError("type must be 'cifar10' or 'cifar100'")

    # Step 2: Organize indices by class
    class_indices = defaultdict(list)
    if hasattr(data, "targets"):
        targets = data.targets
        for idx, label in enumerate(targets):
            class_indices[int(label)].append(idx)
    else:
        for idx in range(len(data)):
            _, label = data[idx]
            class_indices[int(label)].append(idx)

    # Step 3: Class-balanced split
    client_indices = [[] for _ in range(num)]

    for c in range(num_classes):
        idxs = np.array(class_indices[c], dtype=int)
        rng.shuffle(idxs)

        n = len(idxs)
        base = n // num
        rem = n % num

        # 让前 rem 个客户端多拿 1 个（总量尽量均衡）
        start = 0
        for i in range(num):
            take = base + (1 if i < rem else 0)
            if take > 0:
                client_indices[i].extend(idxs[start:start + take].tolist())
            start += take

    sizes = [len(ci) for ci in client_indices]
    if min(sizes) < min_size:
        raise ValueError(
            f"min_size={min_size} not satisfied. got min={min(sizes)}. "
            f"Try smaller num or smaller min_size."
        )

    # Step 4: Create Subset list
    sub_datasets = [Subset(data, np.array(client_indices[i], dtype=int)) for i in range(num)]
    return sub_datasets


class RandomPopList:
    def __init__(self, n: int):
        self.n = n
        self._reset_list()

    def _reset_list(self):
        self.data = list(range(self.n))
        random.shuffle(self.data)  # 随机打乱

    def get(self):
        if not self.data:
            self._reset_list()
        return self.data.pop()


