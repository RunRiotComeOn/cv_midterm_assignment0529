# 基于 MMDetection 的目标检测与实例分割

## 项目概述

本项目为计算机视觉课程（DATA130051.01，复旦大学，2025年春季）中期作业（任务 2）的实现部分。目标是使用 MMDetection 框架，在 PASCAL VOC2012 数据集上训练并评估 Mask R-CNN 和 Sparse R-CNN 两个模型，聚焦于目标检测和实例分割任务。项目对比了两个模型在测试集及 VOC 数据集外的性能，包含数据准备、模型训练、测试及结果可视化的完整流程。

## 目录结构

```raw
task2
├── README.md
├── ckpts
├── config
│   ├── _base_
│   │   ├── datasets
│   │   │   ├── coco_detection.py
│   │   │   └── coco_instance.py
│   │   ├── default_runtime.py
│   │   ├── models
│   │   │   └── mask-rcnn_r50_fpn.py
│   │   └── schedules
│   │       └── schedule_1x.py
│   ├── mask_rcnn
│   │   └── mask-rcnn_r50_fpn_1x_coco.py
│   └── sparse_rcnn
│       └── sparse-rcnn_r50_fpn_1x_coco.py
├── data
│   ├── convert.py
│   ├── download.py
│   └── split.py
├── tools
│   ├── test.py
│   ├── train.py
│   └── visulization.py
└── work_dirs
    ├── mask-rcnn_r50_fpn_1x_coco
    │   └── visualizations
    │       ├── in
    │       │   ├── 2007_007007.jpg
    │       │   ├── 2007_007098.jpg
    │       │   ├── 2009_004901.jpg
    │       │   ├── 2010_000639.jpg
    │       │   ├── a car.jpg
    │       │   ├── a cat.jpg
    │       │   └── street.jpg
    │       └── out
    │           ├── predictions
    │           │   ├── 2007_007007.jpg_predictions.jpg
    │           │   ├── 2007_007098.jpg_predictions.jpg
    │           │   ├── 2009_004901.jpg_predictions.jpg
    │           │   ├── 2010_000639.jpg_predictions.jpg
    │           │   ├── a car.jpg_predictions.jpg
    │           │   ├── a cat.jpg_predictions.jpg
    │           │   └── street.jpg_predictions.jpg
    │           └── proposals
    │               ├── 2007_007007.jpg_proposals.jpg
    │               ├── 2007_007098.jpg_proposals.jpg
    │               ├── 2009_004901.jpg_proposals.jpg
    │               ├── 2010_000639.jpg_proposals.jpg
    │               ├── a car.jpg_proposals.jpg
    │               ├── a cat.jpg_proposals.jpg
    │               └── street.jpg_proposals.jpg
    └── sparse-rcnn_r50_fpn_1x_coco
        └── visualizations
            ├── in
            │   ├── 2007_007007.jpg
            │   ├── 2007_007098.jpg
            │   ├── 2009_004901.jpg
            │   ├── 2010_000639.jpg
            │   ├── a car.jpg
            │   ├── a cat.jpg
            │   └── street.jpg
            └── out
                └── predictions
                    ├── 2007_007007.jpg_predictions.jpg
                    ├── 2007_007098.jpg_predictions.jpg
                    ├── 2009_004901.jpg_predictions.jpg
                    ├── 2010_000639.jpg_predictions.jpg
                    ├── a car.jpg_predictions.jpg
                    ├── a cat.jpg_predictions.jpg
                    └── street.jpg_predictions.jpg

```

- `README.md`：本文件，提供项目概述及使用说明。
- `ckpts/`：存放预训练模型权重的目录。
  - `mask_ckpt.pth`：Mask R-CNN 的预训练权重。
  - `sparse_ckpt.pth`：Sparse R-CNN 的预训练权重。
- `config/`：MMDetection 的配置文件目录。
  - `_base_/`：基础配置文件，包含数据集、模型、训练计划和运行时设置。
    - `datasets/coco_detection.py`：COCO 数据集目标检测配置。
    - `datasets/coco_instance.py`：COCO 数据集实例分割配置。
    - `models/mask-rcnn_r50_fpn.py`：Mask R-CNN 模型基础配置。
    - `schedules/schedule_1x.py`：训练计划基础配置（1x 训练时长）。
    - `default_runtime.py`：默认运行时配置。
  - `mask_rcnn/`：Mask R-CNN 专用配置。
    - `mask-rcnn_r50_fpn_1x_coco.py`：Mask R-CNN 在 COCO 数据集上的完整配置。
  - `sparse_rcnn/`：Sparse R-CNN 专用配置。
    - `sparse-rcnn_r50_fpn_1x_coco.py`：Sparse R-CNN 在 COCO 数据集上的完整配置。
- `data/`：数据处理相关脚本。
  - `convert.py`：将 VOC 格式转换为 COCO 格式的脚本。
  - `download.py`：下载数据集的脚本。
  - `split.py`：划分训练集、验证集和测试集的脚本。
- `tools/`：MMDetection 工具脚本。
  - `train.py`：训练模型的脚本。
  - `test.py`：测试模型的脚本。
  - `visualization.py`：可视化检测结果的脚本（使用 `ckpts/` 中的权重生成边界框和识别结果）。
- `work_dirs/`：训练和测试的输出目录。
  - `mask-rcnn_r50_fpn_1x_coco/`：Mask R-CNN 的输出目录。
    - `visualizations/`：可视化结果。
      - `in/`：输入图像。
        - `2007_007007.jpg`, `2007_007098.jpg`, `2009_004901.jpg`, `2010_000639.jpg`：VOC 数据集测试图像。
        - `a car.jpg`, `a cat.jpg`, `street.jpg`：VOC 数据集外的测试图像。
      - `out/`：输出结果。
        - `proposals/`：Mask R-CNN 的候选框预测图像（如 `a car.jpg_proposals.jpg`）。
        - `predictions/`：Mask R-CNN 的最终预测图像（如 `a car.jpg_predictions.jpg`）。
  - `sparse-rcnn_r50_fpn_1x_coco/`：Sparse R-CNN 的输出目录。
    - `visualizations/`：可视化结果。
      - `in/`：输入图像（与 Mask R-CNN 相同）。
      - `out/`：输出结果。
        - `predictions/`：Sparse R-CNN 的最终预测图像（如 `a cat.jpg_predictions.jpg`）。

## 环境要求

- **操作系统**：Ubuntu 20.04 或以上
- **硬件**：Nvidia RTX4060 GPU（驱动版本：572.83，CUDA 版本：12.8）
- **软件依赖**：
  - Python 3.8+
  - PyTorch 1.10.0+cu113
  - MMDetection 2.25.0
  - MMCV 1.5.0
  - 其他依赖：`numpy`, `matplotlib`, `tqdm`

## 安装步骤

1. **克隆仓库**：

   ```bash
   git clone https://github.com/RunRiotComeOn/cv_midterm_assignment052
   cd task2
   ```

2. **安装依赖**：

   ```bash
   pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
   pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
   pip install mmdet==2.25.0
   pip install numpy matplotlib tqdm
   ```

## 使用说明

### 1. 数据准备

- **下载数据集**：运行以下命令下载 PASCAL VOC2012 数据集：

  ```bash
  python data/download.py
  ```

- **划分数据集**：将数据集划分为训练集、验证集和测试集：

  ```bash
  python data/split.py
  ```

  数据将按 8:1:1 的比例重新划分，生成对应的 `train_new.txt`、`val_new.txt` 和 `test_new.txt` 文件。

- **格式转换**：将 VOC 格式转换为 COCO 格式：

  ```bash
  python data/convert.py
  ```

  输出文件将保存在 `/home/huangyixu/data/VOCdevkit/VOC2012/annotations/` 目录下，包括 `annotations_train.json`、`annotations_val.json` 和 `annotations_test.json`。

### 2. 训练模型

- **Mask R-CNN**：

  ```bash
  python tools/train.py config/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py
  ```

  训练输出（日志、模型权重）将保存在 `work_dirs/mask-rcnn_r50_fpn_1x_coco/`。

- **Sparse R-CNN**：

  ```bash
  python tools/train.py config/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py
  ```

  训练输出将保存在 `work_dirs/sparse-rcnn_r50_fpn_1x_coco/`。

### 3. 测试模型

你可以下载作者训练好的模型至`ckpts/`目录：

- [Mask R-CNN](https://pan.baidu.com/s/1kRg4jLmr_WHFjuLuaOEjOA?pwd=mask)
- [Sparse R-CNN](https://pan.baidu.com/s/18b7iChanoNL9GHHM3w1GYQ?pwd=spar)

- **Mask R-CNN**：

  ```bash
  python tools/test.py config/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py ckpts/mask_ckpt.pth
  ```

- **Sparse R-CNN**：

  ```bash
  python tools/test.py config/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py ckpts/sparse_ckpt.pth
  ```

### 4. 可视化结果

使用 `visualization.py` 脚本生成边界框和识别结果，基于预训练权重 `mask_ckpt.pth` 和 `sparse_ckpt.pth`：

- **Mask R-CNN**：

  ```bash
  python tools/visualization.py config/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py ckpts/mask_ckpt.pth
  ```

  输出图像将保存在 `work_dirs/mask-rcnn_r50_fpn_1x_coco/visualizations/out/`，包括候选框（`proposals/`）和最终预测（`predictions/`）。

- **Sparse R-CNN**：

  ```bash
  python tools/visualization.py config/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py ckpts/sparse_ckpt.pth
  ```

  输出图像将保存在 `work_dirs/sparse-rcnn_r50_fpn_1x_coco/visualizations/out/predictions/`。

## 可视化结果说明

- **输入图像**：`work_dirs/*/visualizations/in/` 包含测试图像，来源于 VOC 数据集（`2007_007007.jpg` 等）和 VOC 数据集外（`a car.jpg` 等）。
- **Mask R-CNN 输出**：
  - `proposals/`：展示候选框预测结果（如 `a car.jpg_proposals.jpg`），反映模型的区域提议能力。
  - `predictions/`：展示最终检测和分割结果（如 `a cat.jpg_predictions.jpg`），包括边界框和掩码。
- **Sparse R-CNN 输出**：
  - `predictions/`：展示最终检测结果（如 `street.jpg_predictions.jpg`），仅包含边界框。

## 注意事项

- 确保 GPU 驱动和 CUDA 版本与环境兼容，否则可能导致运行失败。
- 数据预处理（`convert.py`）可能因文件路径或格式问题失败，请检查 VOC 数据集路径和文件完整性。
- 可视化脚本需要 `matplotlib`，请确保已安装。
- 训练和测试可能因硬件性能不同而耗时较长，建议使用 GPU 加速。

