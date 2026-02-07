import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def test_cifar10_accuracy(model, batch_size=1024, device=None):
    """
    测试模型在 CIFAR-10 测试集上的准确率

    参数:
        model: 已训练的 PyTorch 模型
        batch_size: 测试集 batch size，默认128
        device: 设备 (cuda 或 cpu)，默认自动选择

    返回:
        accuracy: float，模型在测试集上的准确率（0~1之间）
    """
    if device is None:
        device = 'cuda:1'

    model.to(device)
    model.eval()

    # CIFAR-10 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
                             (0.24703224003314972, 0.24348513782024384, 0.26158785820007324))  # 官方推荐均值和方差
    ])

    testset = torchvision.datasets.CIFAR10(root='/home/Suyilin/project/cam/data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def test_cifar100_accuracy(model, batch_size=1024, device=None):
    """
    测试模型在 CIFAR-100 测试集上的准确率

    参数:
        model: 已训练的 PyTorch 模型
        batch_size: 测试集 batch size，默认1024
        device: 设备 (cuda 或 cpu)，默认自动选择

    返回:
        accuracy: float，模型在测试集上的准确率（0~1之间）
    """
    if device is None:
        device = 'cuda:1'

    model.to(device)
    model.eval()

    # CIFAR-100 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071598291397095, 0.4866936206817627, 0.44120192527770996),  # CIFAR-100 均值
                             (0.2673342823982239, 0.2564384639263153, 0.2761504650115967))  # CIFAR-100 标准差
    ])

    testset = torchvision.datasets.CIFAR100(root='/home/Suyilin/project/cam/data', train=False,
                                            download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy



def test_stl10_accuracy(model, batch_size=1024, device=None):
    """
    测试模型在 STL-10 测试集上的准确率

    参数:
        model: 已训练的 PyTorch 模型
        batch_size: 测试集 batch size，默认1024
        device: 设备 (cuda 或 cpu)，默认自动选择

    返回:
        accuracy: float，模型在测试集上的准确率（0~1之间）
    """
    if device is None:
        device = 'cuda:1'

    model.to(device)
    model.eval()

    # STL-10 数据加载，官方推荐大小为96x96
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.44671064615249634, 0.4398098886013031, 0.4066464304924011),  # STL-10 官方均值
                             (0.26034098863601685, 0.2565772831439972, 0.2712673842906952))  # STL-10 官方标准差
    ])

    testset = torchvision.datasets.STL10(
    root='/home/Suyilin/paper_code/grad_cam_plus/data', split='test', download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

