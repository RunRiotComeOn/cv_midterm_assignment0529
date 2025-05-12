# 基于迁移学习的 Caltech-101 图像分类比较实验

本项目实现了一个基于 ResNet-18 的 Caltech-101 图像分类模型，比较了迁移学习（预训练微调）和从零训练的性能，使用 PyTorch 框架，支持 GPU（如 RTX 4060）或 CPU 运行。项目包括数据加载、模型训练、测试、超参数搜索、TensorBoard 可视化以及箱线图分析功能，模块化设计清晰，支持用户自定义数据集路径、输出路径和超参数配置。

## 项目结构

项目包含以下文件：

- `train_and_evaluate.py`: 实现 Caltech-101 数据加载、ResNet-18 模型训练和测试逻辑，包括预训练微调和从零训练实验。
- `boxplot.py`: 提取 TensorBoard 日志，生成验证准确率的箱线图，分析学习率、训练轮数、批次大小和权重衰减的影响。
- `run.sh`: Bash 脚本，用于自动化运行训练和可视化流程，用户只需指定数据集路径和输出路径即可完成整个实验。
- `requirements.txt`: 项目依赖列表，包含所需 Python 包及其版本。

## 环境要求

- **操作系统**: Linux（推荐 Ubuntu）、Windows 或 macOS（支持 Bash 或兼容 shell）。
- **Python 版本**: Python 3.8+
- **依赖库**:
  - 详见 `requirements.txt`，包括 `torch`、`torchvision`、`numpy`、`pandas`、`matplotlib`、`tqdm`、`scikit-learn` 和 `tensorboard`。
- **硬件**: 支持 CUDA 的 NVIDIA GPU（如 RTX 4060）或 CPU。
- **数据集**: Caltech-101 数据集（需下载并解压到指定目录）。

安装依赖的示例命令：
```bash
pip install -r requirements.txt
```

如果使用 GPU，确保安装与 CUDA 兼容的 PyTorch 版本，例如：
```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## 数据准备

1. 下载 Caltech-101 数据集：
   - 官方网站：[Caltech-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
   - 下载链接：http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
   - 或使用脚本自动下载（需安装 `wget` 和 `tar`）：
     ```bash
     mkdir -p ./data
     cd ./data
     wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
     tar -xzvf 101_ObjectCategories.tar.gz
     cd ../..
     ```
   - 解压后，目录结构应包含 `./data/101_ObjectCategories/`，内有各个类别文件夹（如 `accordion/`、`airplanes/` 等）。

2. 确保数据集路径正确， `./data/101_ObjectCategories/`。

## 使用说明

### 1. 训练模型

本项目通过 `run.sh` 脚本简化了训练和可视化流程。用户需指定以下路径和参数：
- `data_root`: Caltech-101 数据集所在的目录（包含 `101_ObjectCategories` 文件夹）。
- `output_dir`: 保存训练日志（TensorBoard 格式）和箱线图的目录。
- 超参数（默认值可通过脚本调整）：学习率 `[0.01, 0.001, 0.0001]`、训练轮数 `[10, 20, 30]`、批次大小 `[16, 32, 64]`、权重衰减 `[0.0001, 0.001]`。

#### 步骤

1. **克隆代码仓库**：
   ```bash
   git clone https://github.com/RunRiotComeOn/cv_midterm_assignment0529.git
   cd cv_midterm_assignment0529
   ```

2. **安装依赖**：
   使用 `requirements.txt` 安装所需库：
   ```bash
   pip install -r requirements.txt
   ```

3. **确保文件权限**：
   为 `run.sh` 添加执行权限：
   ```bash
   chmod +x run.sh
   ```

4. **运行训练**：
   执行以下命令，替换路径为实际路径：
   ```bash
   ./run.sh
   ```
   - 默认使用 `./data` 作为数据集路径，`runs` 作为日志目录，`plots` 作为输出目录。
   - 示例（自定义路径）：
     ```bash
     ./run.sh --data_root /path/to/caltech-101 --output_dir /path/to/plots
     ```
   - 脚本会自动创建 `runs` 和 `plots` 目录（如果不存在），运行预训练和从零训练实验，并生成箱线图。

5. **训练输出**：
   - **控制台**：显示每个 epoch 的训练损失、训练准确率、验证损失和验证准确率，以及最佳模型和从零训练的结果。
   - **日志目录**（`runs`）：
     - TensorBoard 日志文件，用于可视化训练过程。
   - **输出目录**（`plots`）：
     - `val_acc_boxplot_lr.png`: 学习率箱线图。
     - `val_acc_boxplot_epochs.png`: 训练轮数箱线图。
     - `val_acc_boxplot_bs.png`: 批次大小箱线图。
     - `val_acc_boxplot_wd.png`: 权重衰减箱线图。
   - **模型文件**：
     - `best_pretrained_model.pth`: 最佳预训练模型权重，保存在当前目录。

### 2. 测试模型

测试过程已集成在 `train_and_evaluate.py` 中，脚本会自动在训练后评估最佳预训练模型和从零训练模型，无需单独运行测试脚本。

如果需要手动测试模型权重，可以编写单独脚本，加载 `best_pretrained_model.pth` 并调用 `test_model` 函数。可以参考这个脚本：

```python
import torch
from torchvision.models import resnet18
from train_and_evaluate import test_model, load_and_preprocess_data

# 加载数据
train_loaders, test_loaders, _, _ = load_and_preprocess_data(data_root='./data', batch_sizes=[16])

# 初始化模型
model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 101)
model.load_state_dict(torch.load('best_pretrained_model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 测试
acc = test_model(model, test_loaders[16], torch.nn.CrossEntropyLoss(), device)
print(f"Test Accuracy: {acc:.2f}%")
```

### 3. 模型权重下载

训练好的模型权重（`best_pretrained_model.pth`）可通过实验生成，或从以下链接下载预训练权重：

- **百度云盘链接**:
  [https://pan.baidu.com/s/1kz_0LN4F4N37a-agWmtqIg?pwd=best](https://pan.baidu.com/s/1kz_0LN4F4N37a-agWmtqIg?pwd=best)  
  **提取码**: best

**说明**：
- 下载后将权重文件放置在项目目录中，确保路径与脚本一致。
- 最佳模型权重基于 `lr=0.01, epochs=30, bs=16, wd=0.001`（实验结果）。

### 4. 可视化与分析

#### TensorBoard 可视化

TensorBoard 可用于查看训练过程中的损失和准确率曲线，具体步骤如下：

1. **启动 TensorBoard**：
   在项目目录下运行以下命令，指定日志目录 `runs`：
   ```bash
   tensorboard --logdir=runs --host 0.0.0.0 --port 6006
   ```

2. **访问 TensorBoard**：
   - 如果在本地运行，打开浏览器访问：
     ```
     http://localhost:6006
     ```
   - 如果在远程服务器运行，使用 `--host 0.0.0.0` 后，可以通过服务器 IP 访问，例如：
     ```
     http://<server-ip>:6006
     ```
   - 您也可以通过 SSH 端口转发访问：
     ```bash
     ssh -L 6006:localhost:6006 <username>@<server-ip>
     ```
     然后在本地浏览器访问 `http://localhost:6006`。

3. **查看内容**：
   - **Scalars** 页面：显示训练损失（`Loss/Train`）、验证损失（`Loss/Val`）和验证准确率（`Accuracy/Val`）曲线。
   - 每个实验运行（如 `Pretrained_lr_0.01_epochs_30_bs_16_wd_0.001`）对应一个独立曲线。

#### 箱线图分析

- `plots` 目录下的箱线图展示了不同超参数组合的验证准确率分布，帮助优化超参数选择：
  - `val_acc_boxplot_lr.png`: 学习率对验证准确率的影响。
  - `val_acc_boxplot_epochs.png`: 训练轮数对验证准确率的影响。
  - `val_acc_boxplot_bs.png`: 批次大小对验证准确率的影响。
  - `val_acc_boxplot_wd.png`: 权重衰减对验证准确率的影响。

### 5. 超参数搜索与分析

超参数搜索已在 `train_and_evaluate.py` 中实现，搜索范围包括：
- 学习率：`[0.01, 0.001, 0.0001]`
- 训练轮数：`[10, 20, 30]`
- 批次大小：`[16, 32, 64]`
- 权重衰减：`[0.0001, 0.001]`

**结果分析**：
- 预训练模型最佳准确率 97.21%（`lr=0.01, epochs=30, bs=16, wd=0.001`）。
- 从零训练最高准确率 79.0%（`lr=0.001, epochs=30, bs=16, wd=0.001`）。
- 箱线图（`plots` 目录）显示学习率对性能影响最大，`lr=0.01` 适合预训练，而 `lr=0.001` 更适合从零训练。

## 常见问题

1. **Q: 数据集路径错误怎么办？**
   - A: 确保 `data_root` 指向正确的 Caltech-101 目录，包含 `101_ObjectCategories` 文件夹。检查路径是否拼写正确。

2. **Q: GPU 报错“CUDA 环境不兼容”？**
   - A: 确认安装的 PyTorch 版本与您的 CUDA 版本匹配。例如，RTX 4060 通常支持 CUDA 12.x，使用 `pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121` 安装。

3. **Q: 模型权重文件无法加载？**
   - A: 确保 `.pth` 文件路径正确，且文件未损坏。尝试重新训练或下载。

4. **Q: 如何修改超参数范围？**
   - A: 编辑 `train_and_evaluate.py` 中的 `learning_rates`、`num_epochs_list`、`batch_sizes` 和 `weight_decays` 列表，添加或修改值。

5. **Q: TensorBoard 无法访问？**
   - A: 确保端口 6006 未被占用，尝试更换端口（例如 `--port 6007`）。如果在远程服务器上运行，确保网络安全组或防火墙允许访问，或者使用 SSH 端口转发。

## 联系方式

如有问题，请通过 GitHub Issues 提交，或联系 [23307110412@m.fudan.edu.cn](mailto:23307110412@m.fudan.edu.cn)
