import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Caltech101
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data(data_root='./data', batch_sizes=[16, 32, 64]):
    """
    加载并预处理 Caltech-101 数据集，按照 80% 训练集和 20% 测试集划分。

    参数：
    - data_root (str): 数据集保存路径，默认为 './data'
    - batch_sizes (list): 批次大小列表，默认为 [16, 32, 64]

    返回：
    - train_loaders (dict): 训练数据加载器，按 batch_size 索引
    - test_loaders (dict): 测试数据加载器，按 batch_size 索引
    - train_dataset (Subset): 训练数据集
    - test_dataset (Subset): 测试数据集
    """
    print("=== Loading and Preprocessing Dataset ===")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Caltech101(root=data_root, download=False, transform=transform)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loaders = {bs: DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2) for bs in batch_sizes}
    test_loaders = {bs: DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2) for bs in batch_sizes}

    print(f"Train Set Size: {len(train_dataset)}, Test Set Size: {len(test_dataset)}")
    return train_loaders, test_loaders, train_dataset, test_dataset

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, writer, run_name):
    """
    训练模型，并使用 TensorBoard 记录损失和准确率。

    参数：
    - model: 模型实例
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs (int): 训练轮数
    - device: 设备（CPU 或 GPU）
    - writer: SummaryWriter 实例，用于 TensorBoard 记录
    - run_name (str): 实验运行名称
    """
    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            global_step += 1
            writer.add_scalar(f'Loss/Train/{run_name}', loss.item(), global_step)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        writer.add_scalar(f'Loss/Val/{run_name}', val_loss, epoch + 1)
        writer.add_scalar(f'Accuracy/Val/{run_name}', val_acc, epoch + 1)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

def evaluate_model(model, test_loader, criterion, device):
    """
    评估模型在测试集上的损失和准确率。

    参数：
    - model: 模型实例
    - test_loader: 测试数据加载器
    - criterion: 损失函数
    - device: 设备（CPU 或 GPU）

    返回：
    - test_loss (float): 测试损失
    - test_acc (float): 测试准确率（百分比）
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    model.train()
    return test_loss, test_acc

def test_model(model, test_loader, criterion, device):
    """
    测试模型在测试集上的性能。

    参数：
    - model: 模型实例
    - test_loader: 测试数据加载器
    - criterion: 损失函数
    - device: 设备（CPU 或 GPU）

    返回：
    - test_acc (float): 测试准确率（百分比）
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_acc

def run_experiments(train_loaders, test_loaders, learning_rates, num_epochs_list, batch_sizes, weight_decays, device):
    """
    运行实验，包括预训练 ResNet-18 微调和从零训练。

    参数：
    - train_loaders (dict): 训练数据加载器，按 batch_size 索引
    - test_loaders (dict): 测试数据加载器，按 batch_size 索引
    - learning_rates (list): 学习率列表
    - num_epochs_list (list): 训练轮数列表
    - batch_sizes (list): 批次大小列表
    - weight_decays (list): 权重衰减列表
    - device: 设备（CPU 或 GPU）

    返回：
    - best_acc (float): 最佳验证准确率
    - best_params (dict): 最佳超参数组合
    - scratch_acc (float): 从零训练的验证准确率
    """
    best_acc = 0.0
    best_params = {}

    # 实验 1：预训练 ResNet-18 微调
    print("=== Experiment 1: Fine-tuning Pretrained ResNet-18 ===")
    for lr in learning_rates:
        for num_epochs in num_epochs_list:
            for bs in batch_sizes:
                for wd in weight_decays:
                    print(f"\nLearning Rate: {lr}, Epochs: {num_epochs}, Batch Size: {bs}, Weight Decay: {wd}")
                    run_name = f"Pretrained_lr_{lr}_epochs_{num_epochs}_bs_{bs}_wd_{wd}"
                    writer = SummaryWriter(log_dir=f'runs/{run_name}')
                    model = models.resnet18(pretrained=True)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, 101)
                    model = model.to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD([
                        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': lr * 0.1},
                        {'params': model.fc.parameters(), 'lr': lr}
                    ], momentum=0.9, weight_decay=wd)
                    train_model(model, train_loaders[bs], test_loaders[bs], criterion, optimizer, num_epochs, device, writer, run_name)
                    acc = test_model(model, test_loaders[bs], criterion, device)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = {'lr': lr, 'epochs': num_epochs, 'batch_size': bs, 'weight_decay': wd}
                        torch.save(model.state_dict(), 'best_pretrained_model.pth')
                    writer.close()

    print(f"\nBest Pretrained Model Accuracy: {best_acc:.2f}% with params: {best_params}")

    # 实验 2：从头训练 ResNet-18
    print("\n=== Experiment 2: Training ResNet-18 from Scratch ===")
    run_name = f"Scratch_lr_{best_params['lr']}_epochs_{best_params['epochs']}_bs_{best_params['batch_size']}_wd_{best_params['weight_decay']}"
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    model_scratch = models.resnet18(pretrained=False)
    num_ftrs = model_scratch.fc.in_features
    model_scratch.fc = nn.Linear(num_ftrs, 101)
    model_scratch = model_scratch.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_scratch.parameters(), lr=best_params['lr'], momentum=0.9, weight_decay=best_params['weight_decay'])
    train_model(model_scratch, train_loaders[best_params['batch_size']], test_loaders[best_params['batch_size']], criterion, optimizer, best_params['epochs'], device, writer, run_name)
    scratch_acc = test_model(model_scratch, test_loaders[best_params['batch_size']], criterion, device)
    writer.close()

    print(f"\nScratch Model Accuracy: {scratch_acc:.2f}%")
    print(f"Accuracy Improvement from Pretraining: {best_acc - scratch_acc:.2f}%")

    return best_acc, best_params, scratch_acc

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train ResNet-18 on Caltech-101 dataset with pretraining and scratch experiments.")
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for Caltech-101 dataset (default: ./data)')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[0.01, 0.001, 0.0001], help='Learning rates to test (default: [0.01, 0.001, 0.0001])')
    parser.add_argument('--num_epochs_list', type=int, nargs='+', default=[10, 20, 30], help='Number of epochs to test (default: [10, 20, 30])')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16, 32, 64], help='Batch sizes to test (default: [16, 32, 64])')
    parser.add_argument('--weight_decays', type=float, nargs='+', default=[1e-4, 1e-3], help='Weight decays to test (default: [1e-4, 1e-3])')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载和预处理数据
    train_loaders, test_loaders, _, _ = load_and_preprocess_data(data_root=args.data_root, batch_sizes=args.batch_sizes)

    # 运行实验
    best_acc, best_params, scratch_acc = run_experiments(
        train_loaders,
        test_loaders,
        args.learning_rates,
        args.num_epochs_list,
        args.batch_sizes,
        args.weight_decays,
        device
    )

    print("\n=== Experiment Summary ===")
    print(f"Best Pretrained Model Accuracy: {best_acc:.2f}% with params: {best_params}")
    print(f"Scratch Model Accuracy: {scratch_acc:.2f}%")
    print(f"Accuracy Improvement from Pretraining: {best_acc - scratch_acc:.2f}%")

if __name__ == "__main__":
    main()