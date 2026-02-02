import subprocess
import sys
import shutil
import importlib.util

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

def check_nvidia_gpu():
    """檢查系統中是否有 NVIDIA 顯卡 (透過 nvidia-smi)"""
    print("\n🔍 正在檢查 NVIDIA 顯卡...")
    # 檢查 nvidia-smi 指令是否存在
    if shutil.which("nvidia-smi") is not None:
        try:
            # 執行 nvidia-smi 確認驅動是否正常運作
            subprocess.check_output(["nvidia-smi"])
            print("✅ 偵測到 NVIDIA 顯卡與驅動程式！")
            return True
        except subprocess.CalledProcessError:
            print("⚠️ 偵測到 nvidia-smi 但無法執行，將視為無顯卡模式。")
            return False
    else:
        print("⚠️ 未偵測到 NVIDIA 顯卡，將使用 CPU 模式。")
        return False

def uninstall_torch():
    """強制移除現有的 torch 相關套件，防止版本衝突"""
    print("\n🧹 清理舊版 torch 函式庫 (防止衝突)...")
    pkgs = ["torch", "torchvision", "torchaudio"]
    # 為了避免找不到套件報錯，我們逐一嘗試移除
    cmd = ["uninstall", "-y"] + pkgs
    try:
        subprocess.call([sys.executable, "-m", "pip"] + cmd)
        print("✅ 舊版清理完成")
    except Exception as e:
        print(f"⚠️ 清理過程略過: {e}")

def install_torch(has_gpu):
    """根據是否有 GPU 安裝對應版本的 PyTorch"""
    if has_gpu:
        print("\n🚀 正在下載並安裝 PyTorch (CUDA 12.6 版本)...")
        print("這可能需要一段時間，請保持網路連線...")
        # PyTorch 官方 CUDA 12.6 安裝指令
        # 注意: 如果官方尚未完全釋出 12.6 穩定版，pip 可能會自動退回 12.4 或 12.1，但我們會指定 index-url
        run_pip(
            [
                "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu126"
            ],
            "PyTorch (CUDA 12.6)"
        )
    else:
        print("\n🐢 正在下載並安裝 PyTorch (CPU 版本)...")
        run_pip(
            ["install", "torch", "torchvision", "torchaudio"],
            "PyTorch (CPU)"
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
            print(f"✅ CUDA 狀態: 可用")
            print(f"✅ 當前 CUDA 版本: {torch.version.cuda}")
            print(f"✅ 顯卡型號: {torch.cuda.get_device_name(0)}")
            print("\n🎉 環境建置成功！你的模型將會在 GPU 上奔跑！")
        else:
            print("⚠️ CUDA 狀態: 不可用 (將使用 CPU 訓練)")
            print("\n🎉 環境建置成功 (CPU 模式)。")
            
    except ImportError:
        print("❌ 驗證失敗：無法匯入 torch，請檢查安裝過程。")

def main():
    print("========================================")
    print("      SCNN 專案自動環境建置工具")
    print("========================================")
    
    # 1. 檢查顯卡
    has_gpu = check_nvidia_gpu()
    
    # 2. 移除舊版防止衝突
    uninstall_torch()
    
    # 3. 安裝 PyTorch
    install_torch(has_gpu)
    
    # 4. 安裝其他依賴
    install_other_dependencies()
    
    # 5. 驗證
    verify_installation()
    
    input("\n按 Enter 鍵退出程式...")

if __name__ == "__main__":
    main()