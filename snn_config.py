# snn_config.py
import torch
import os

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

# --- [修改] 權重檔案路徑設定 ---
WEIGHT_DIR = "./WEIGHT"  # 設定權重存放目錄
# 確保目錄存在
if not os.path.exists(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)
# 預設自動產生的檔名 (現在會包含路徑)
WEIGHTS_FILE = os.path.join(WEIGHT_DIR, f"{DATASET_NAME}_{MODEL_ARCH}.csv")

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


# ==========================================
# [新增] 類別名稱對照表 (從 GUI 移過來)
# ==========================================
CLASS_LABELS = {
    "MNIST": [str(i) for i in range(10)], # 0, 1, 2...
    "CIFAR10": ["飛機", "汽車", "鳥", "貓", "鹿", "狗", "青蛙", "馬", "船", "卡車"]
}

# ==========================================
# [新增] 預設參數設定檔 (從 GUI 移過來)
# ==========================================
PRESETS = {
    "MNIST": {
        "arch": "6C5-PL-16C5-PL-FC120-FC84-FC10",
        "epochs": 3,
        "timesteps": 20,
        "beta": 0.95,
        "lr": 0.001,
        "threshold": 1.0,
        "batch": 128
    },
    "CIFAR10": {
        "arch": "16C3-64C3-PL-128C3-128C3-PL-128C3-PL-FC512-FC10",
        "epochs": 10,
        "timesteps": 30,
        "beta": 0.90,
        "lr": 0.001,
        "threshold": 1.0,
        "batch": 64
    }
}