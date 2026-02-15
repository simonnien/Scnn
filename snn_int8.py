import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import snntorch as snn
import snn_config as cfg

class INT8SimulatedConv2d(nn.Module):
    def __init__(self, orig_conv):
        super().__init__()
        self.stride = orig_conv.stride
        self.padding = orig_conv.padding
        self.dilation = orig_conv.dilation
        self.groups = orig_conv.groups
        
        max_val = torch.max(torch.abs(orig_conv.weight.data))
        self.scale = max_val / 127.0 if max_val > 0 else 1.0
        
        self.weight_int8 = torch.round(orig_conv.weight.data / self.scale)
        if orig_conv.bias is not None:
            self.bias_int8 = torch.round(orig_conv.bias.data / self.scale)
        else:
            self.bias_int8 = None

    def forward(self, x):
        out_int32 = F.conv2d(x, self.weight_int8, self.bias_int8, 
                             self.stride, self.padding, self.dilation, self.groups)
        return out_int32 * self.scale

class INT8SimulatedLinear(nn.Module):
    def __init__(self, orig_linear):
        super().__init__()
        max_val = torch.max(torch.abs(orig_linear.weight.data))
        self.scale = max_val / 127.0 if max_val > 0 else 1.0
        
        self.weight_int8 = torch.round(orig_linear.weight.data / self.scale)
        if orig_linear.bias is not None:
            self.bias_int8 = torch.round(orig_linear.bias.data / self.scale)
        else:
            self.bias_int8 = None

    def forward(self, x):
        out_int32 = F.linear(x, self.weight_int8, self.bias_int8)
        return out_int32 * self.scale

# === [關鍵修正] 繼承 snn.Leaky 並解決 GPU Tensor 轉 NumPy 問題 ===
class INT8Leaky(snn.Leaky):
    def __init__(self, orig_leaky, orig_thresh):
        # 1. 將 GPU 上的 Tensor 取出為純數字 (Float)，避免 Numpy 報錯
        beta_val = orig_leaky.beta.item() if isinstance(orig_leaky.beta, torch.Tensor) else orig_leaky.beta
        
        # 2. 呼叫父類別 (snn.Leaky) 的初始化，讓 snn_model.py 認得它是 Leaky 層
        super().__init__(beta=beta_val, threshold=orig_thresh)
        
        self.orig_thresh = orig_thresh
        self.int_thresh = 127 
        
        # 3. 使用純數字 beta_val 進行對數運算
        if beta_val >= 1.0:
            self.shift = 100 
        else:
            self.shift = max(1, int(np.round(-np.log2(1.0 - beta_val))))
            
    def init_leaky(self):
        return torch.tensor(0.0, device=cfg.DEVICE)

    def forward(self, x, mem):
        scale_to_int = 127.0 / self.orig_thresh
        x_int = torch.round(x * scale_to_int)
        
        if self.shift < 100:
            decay = torch.floor(mem / (2 ** self.shift))
            decayed_mem = mem - decay
        else:
            decayed_mem = mem
            
        mem_new = decayed_mem + x_int
        mem_new = torch.clamp(mem_new, -128, 127)
        
        spk = (mem_new >= self.int_thresh).float()
        mem_new = mem_new * (1.0 - spk)
        
        return spk, mem_new

def convert_to_int8(model):
    """將模型的所有層替換為純 INT8 運算 (包含神經元)"""
    print("\n[INT8] 正在將模型轉換為 100% 全量化 (INT8) 運算模式...")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Conv2d):
            model.layers[i] = INT8SimulatedConv2d(layer)
            print(f"  - L{i:02d} (Conv2d) 轉換 INT8 成功")
        elif isinstance(layer, nn.Linear):
            model.layers[i] = INT8SimulatedLinear(layer)
            print(f"  - L{i:02d} (Linear) 轉換 INT8 成功")
        elif isinstance(layer, snn.Leaky):
            model.layers[i] = INT8Leaky(layer, cfg.THRESHOLD)
            print(f"  - L{i:02d} (Leaky)  轉換 INT8 (位移漏電, 閾值=127) 成功")
    
    print("[INT8] 模型量化轉換完成！")
    return model