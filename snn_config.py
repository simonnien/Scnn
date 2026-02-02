# snn_config.py
import torch

# --- 資料庫設定 [新增] ---
# 支援選項: "MNIST", "CIFAR10"
DATASET_NAME = "MNIST"

# --- 圖像規格 (將由 GUI 自動依據 Dataset 更新) [新增] ---
# MNIST預設: 28x28, 1 Channel
# CIFAR10預設: 32x32, 3 Channels
IMAGE_SIZE = 28
INPUT_CHANNELS = 1

# --- 核心：模型架構定義 ---
MODEL_ARCH = "6C5-PL-16C5-PL-FC120-FC84-FC10"

# --- 自動產生權重檔名 ---
WEIGHTS_FILE = f"{DATASET_NAME}_{MODEL_ARCH}.csv"

# --- A. 訓練相關參數 ---
BATCH_SIZE = 128        # 批次大小
LEARNING_RATE = 1e-3    # 學習率
NUM_EPOCHS = 1          # 訓練輪數
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = './data'

# --- B. SNN/SCNN 物理參數 ---
TIME_STEPS = 25         # 時間步長 (模擬一張圖要用幾個時間點，越高越準但越慢)
BETA = 0.95             # 漏電衰減率 (Decay Rate)，範圍 0~1，越接近 1 記憶越久
THRESHOLD = 1.0         # 脈衝發射閾值
SLOPE = 25              # 代理梯度 (Surrogate Gradient) 的斜率，影響反向傳播
