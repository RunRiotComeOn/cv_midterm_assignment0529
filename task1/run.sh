#!/bin/bash

# run.sh
# 脚本用于运行 Caltech-101 数据集上的 ResNet-18 训练和可视化实验

# 设置日志文件
LOG_FILE="experiment.log"
echo "Starting experiment at $(date)" | tee -a $LOG_FILE

# 定义 Python 解释器（根据您的环境调整）
PYTHON="python3"

# 检查 Python 环境
if ! command -v $PYTHON &> /dev/null; then
    echo "Error: $PYTHON not found. Please ensure Python is installed." | tee -a $LOG_FILE
    exit 1
fi

# 检查依赖文件是否存在
TRAIN_SCRIPT="train_and_evaluate.py"
BOXPLOT_SCRIPT="boxplot.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: $TRAIN_SCRIPT not found." | tee -a $LOG_FILE
    exit 1
fi

if [ ! -f "$BOXPLOT_SCRIPT" ]; then
    echo "Error: $BOXPLOT_SCRIPT not found." | tee -a $LOG_FILE
    exit 1
fi

# 定义实验参数
DATA_ROOT="./data"
LEARNING_RATES="0.01 0.001 0.0001"
NUM_EPOCHS="10 20 30"
BATCH_SIZES="16 32 64"
WEIGHT_DECAYS="0.0001 0.001"
LOG_DIR="runs"
OUTPUT_DIR="plots"

# 确保数据目录存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data directory $DATA_ROOT does not exist. Please ensure Caltech-101 dataset is downloaded." | tee -a $LOG_FILE
    exit 1
fi

# 运行训练和评估
echo "Running training and evaluation..." | tee -a $LOG_FILE
$PYTHON $TRAIN_SCRIPT \
    --data_root "$DATA_ROOT" \
    --learning_rates $LEARNING_RATES \
    --num_epochs_list $NUM_EPOCHS \
    --batch_sizes $BATCH_SIZES \
    --weight_decays $WEIGHT_DECAYS \
    | tee -a $LOG_FILE

if [ $? -ne 0 ]; then
    echo "Error: Training failed." | tee -a $LOG_FILE
    exit 1
fi

# 运行箱线图生成
echo "Generating boxplots..." | tee -a $LOG_FILE
$PYTHON $BOXPLOT_SCRIPT \
    --log_dir "$LOG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_dataset_size 7315 \
    | tee -a $LOG_FILE

if [ $? -ne 0 ]; then
    echo "Error: Boxplot generation failed." | tee -a $LOG_FILE
    exit 1
fi

echo "Experiment completed at $(date)" | tee -a $LOG_FILE
echo "Logs saved to $LOG_FILE"
echo "Boxplots saved to $OUTPUT_DIR"