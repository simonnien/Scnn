import torch

# 1. 檢查 CUDA 是否可用
if torch.cuda.is_available():
    print("✅ 成功抓到顯卡！")
    print(f"顯卡數量: {torch.cuda.device_count()}")
    print(f"顯卡型號: {torch.cuda.get_device_name(0)}")
    print(f"當前 CUDA 版本: {torch.version.cuda}")
else:
    print("❌ 抓不到顯卡，目前只能使用 CPU。")

input("\n按 Enter 鍵退出程式...")
