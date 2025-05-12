import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import argparse

def plot_metrics(log_dir, output_dir='plots', train_dataset_size=7315):
    """
    从 TensorBoard 日志中提取验证准确率，绘制超参数的箱线图。

    参数：
    - log_dir (str): TensorBoard 日志目录
    - output_dir (str): 输出图像保存目录，默认为 'plots'
    - train_dataset_size (int): 训练数据集大小，默认为 7315（Caltech-101）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    val_accs, params = {}, {}

    # 提取验证准确率和超参数
    for run in runs:
        event_acc = EventAccumulator(os.path.join(log_dir, run))
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        
        val_acc_tag = f'Accuracy/Val/{run}'
        if val_acc_tag in tags:
            val_accs[run] = [(e.step, e.value) for e in event_acc.Scalars(val_acc_tag)]
        
        try:
            params[run] = {
                'lr': float(run.split('_lr_')[1].split('_')[0]),
                'epochs': int(run.split('_epochs_')[1].split('_')[0]),
                'bs': int(run.split('_bs_')[1].split('_')[0]),
                'wd': float(run.split('_wd_')[1])
            }
        except IndexError:
            print(f"Warning: Run name {run} does not match expected format, skipping parameter parsing.")
            params[run] = {'lr': 0.0, 'epochs': 0, 'bs': 0, 'wd': 0.0}

    data = {
        'Run': [],
        'Val Accuracy': [],
        'Learning Rate': [],
        'Epochs': [],
        'Batch Size': [],
        'Weight Decay': []
    }
    for run in runs:
        if run in val_accs and val_accs[run]:
            last_acc = val_accs[run][-1][1]
            data['Run'].append(run)
            data['Val Accuracy'].append(last_acc)
            data['Learning Rate'].append(params[run]['lr'])
            data['Epochs'].append(params[run]['epochs'])
            data['Batch Size'].append(params[run]['bs'])
            data['Weight Decay'].append(params[run]['wd'])

    df = pd.DataFrame(data)

    # 设置绘图样式
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['boxplot.boxprops.linewidth'] = 1.5
    plt.rcParams['boxplot.whiskerprops.linewidth'] = 1.5
    plt.rcParams['boxplot.medianprops.linewidth'] = 2

    # 图 1：学习率箱线图
    plt.figure()
    lr_groups = [df[df['Learning Rate'] == lr]['Val Accuracy'].values for lr in sorted(df['Learning Rate'].unique())]
    bp = plt.boxplot(lr_groups, labels=[f'{lr}' for lr in sorted(df['Learning Rate'].unique())], patch_artist=True)
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Set2 调色板颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    plt.title('Validation Accuracy Distribution vs Learning Rate', pad=15)
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_acc_boxplot_lr.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 2：训练轮数箱线图
    plt.figure()
    epochs_groups = [df[df['Epochs'] == epochs]['Val Accuracy'].values for epochs in sorted(df['Epochs'].unique())]
    bp = plt.boxplot(epochs_groups, labels=[f'{epochs}' for epochs in sorted(df['Epochs'].unique())], patch_artist=True)
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Set2 调色板颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    plt.title('Validation Accuracy Distribution vs Epochs', pad=15)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_acc_boxplot_epochs.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 3：批次大小箱线图
    plt.figure()
    bs_groups = [df[df['Batch Size'] == bs]['Val Accuracy'].values for bs in sorted(df['Batch Size'].unique())]
    bp = plt.boxplot(bs_groups, labels=[f'{bs}' for bs in sorted(df['Batch Size'].unique())], patch_artist=True)
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Set2 调色板颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    plt.title('Validation Accuracy Distribution vs Batch Size', pad=15)
    plt.xlabel('Batch Size')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_acc_boxplot_bs.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 4：权重衰减箱线图
    plt.figure()
    wd_groups = [df[df['Weight Decay'] == wd]['Val Accuracy'].values for wd in sorted(df['Weight Decay'].unique())]
    bp = plt.boxplot(wd_groups, labels=[f'{wd}' for wd in sorted(df['Weight Decay'].unique())], patch_artist=True)
    colors = ['#66c2a5', '#fc8d62']  # Set2 调色板颜色（仅两个值）
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    plt.title('Validation Accuracy Distribution vs Weight Decay', pad=15)
    plt.xlabel('Weight Decay')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_acc_boxplot_wd.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Visualize TensorBoard logs and plot validation accuracy boxplots.")
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory containing TensorBoard logs (default: runs)')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save output plots (default: plots)')
    parser.add_argument('--train_dataset_size', type=int, default=7315, help='Size of the training dataset (default: 7315)')
    args = parser.parse_args()

    # 调用可视化函数
    print(f"Processing logs from {args.log_dir}...")
    plot_metrics(args.log_dir, args.output_dir, args.train_dataset_size)
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()