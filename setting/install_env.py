import subprocess
import sys
import shutil
import re

def run_pip(args, description):
    """執行 pip 指令的輔助函數"""
    print(f"\n[正在執行] {description}...")
    try:
        # 使用 sys.executable 確保是安裝在當前執行的 Python 環境中
        cmd = [sys.executable, "-m", "pip"] + args
        subprocess.check_call(cmd)
        print(f"✅ {description} 成功")
    except subprocess.CalledProcessError:
        print(f"❌ {description} 失敗")
        sys.exit(1)

def check_gpu_info():
    """
    檢查系統中是否有 NVIDIA 顯卡，並嘗試取得型號以判斷是否為 RTX 50 系列
    回傳: (has_gpu, is_rtx50_series)
    """
    print("\n🔍 正在檢查 NVIDIA 顯卡...")
    
    has_gpu = False
    is_rtx50 = False
    
    if shutil.which("nvidia-smi") is not None:
        try:
            # 執行 nvidia-smi 取得顯卡名稱
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], encoding='utf-8')
            print(f"✅ 偵測到顯卡: {output.strip()}")
            has_gpu = True
            
            # 判斷是否為 RTX 50 系列 (包含 Laptop GPU)
            if "RTX 50" in output:
                print("⚡ 偵測到 RTX 50 系列顯卡！將啟用 Nightly 版本安裝模式。")
                is_rtx50 = True
            else:
                print("ℹ️ 偵測到一般 NVIDIA 顯卡，將使用 Stable 版本安裝模式。")
                
        except subprocess.CalledProcessError:
            print("⚠️ 偵測到 nvidia-smi 但無法執行，將視為無顯卡模式。")
    else:
        print("⚠️ 未偵測到 NVIDIA 顯卡，將使用 CPU 模式。")
    
    return has_gpu, is_rtx50

def uninstall_torch():
    """強制移除現有的 torch 相關套件，防止版本衝突"""
    print("\n🧹 清理舊版 torch 函式庫 (防止衝突)...")
    pkgs = ["torch", "torchvision", "torchaudio"]
    cmd = ["uninstall", "-y"] + pkgs
    try:
        subprocess.call([sys.executable, "-m", "pip"] + cmd)
        print("✅ 舊版清理完成")
    except Exception as e:
        print(f"⚠️ 清理過程略過: {e}")

def install_torch(has_gpu, is_rtx50):
    """根據顯卡類型安裝對應版本的 PyTorch"""
    
    if not has_gpu:
        print("\n🐢 正在下載並安裝 PyTorch (CPU 版本)...")
        run_pip(
            ["install", "torch", "torchvision", "torchaudio"],
            "PyTorch (CPU)"
        )
        return

    if is_rtx50:
        print("\n🚀 正在安裝 PyTorch Nightly (預覽版) 以支援 RTX 50 系列...")
        print("目標 CUDA 版本: 12.8+ (相容 sm_120 架構)")
        # RTX 50 系列需要 Nightly 版本
        run_pip(
            [
                "install", 
                "--pre", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
            ],
            "PyTorch Nightly (RTX 50 Series Support)"
        )
    else:
        print("\n🚀 正在下載並安裝 PyTorch Stable (穩定版)...")
        print("目標 CUDA 版本: 12.6")
        # 一般顯卡使用穩定的 CUDA 12.6
        run_pip(
            [
                "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu126"
            ],
            "PyTorch Stable (CUDA 12.6)"
        )

def install_other_dependencies():
    """安裝專案所需的其他函式庫"""
    print("\n📦 正在安裝其他專案依賴 (snntorch, pillow, numpy)...")
    requirements = ["snntorch", "pillow", "numpy", "matplotlib"]
    run_pip(["install"] + requirements, "專案依賴套件")

def verify_installation():
    """驗證安裝結果"""
    print("\n🔍 正在驗證安裝結果...")
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            cap = torch.cuda.get_device_capability(0)
            
            print(f"✅ CUDA 狀態: 可用")
            print(f"✅ 當前 CUDA 版本: {cuda_ver}")
            print(f"✅ 顯卡型號: {gpu_name}")
            print(f"✅ 算力架構: {cap[0]}.{cap[1]}")
            
            # 進行簡單運算測試
            try:
                x = torch.tensor([1.0]).cuda()
                print("✅ Tensor 運算測試: 通過")
                print("\n🎉 環境建置成功！你的模型將會在 GPU 上奔跑！")
            except Exception as e:
                print(f"❌ 顯卡已偵測到，但運算失敗: {e}")
                print("建議檢查 NVIDIA 驅動程式是否為最新版。")
        else:
            print("⚠️ CUDA 狀態: 不可用 (將使用 CPU 訓練)")
            print("可能原因：顯卡驅動過舊或 PyTorch 版本不匹配。")
            
    except ImportError:
        print("❌ 驗證失敗：無法匯入 torch，請檢查安裝過程。")
    except Exception as e:
        print(f"❌ 發生未預期錯誤: {e}")

def main():
    print("========================================")
    print("      SCNN 專案自動環境建置工具")
    print("      (支援 RTX 50 系列自動判斷)")
    print("========================================")
    
    # 1. 檢查顯卡型號
    has_gpu, is_rtx50 = check_gpu_info()
    
    # 2. 移除舊版防止衝突
    uninstall_torch()
    
    # 3. 安裝 PyTorch (自動選擇版本)
    install_torch(has_gpu, is_rtx50)
    
    # 4. 安裝其他依賴
    install_other_dependencies()
    
    # 5. 驗證
    verify_installation()
    
    input("\n按 Enter 鍵退出程式...")

if __name__ == "__main__":
    main()
    