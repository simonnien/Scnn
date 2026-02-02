import torch
import torch.nn as nn
import snntorch as snn
import csv
import numpy as np
import ast
import snn_config as cfg

class DynamicSCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch_str = cfg.MODEL_ARCH
        self.layers = nn.ModuleList()
        self.spike_grad = snn.surrogate.fast_sigmoid(slope=cfg.SLOPE)
        
        # [修改] 從 Config 讀取初始通道數與圖片大小
        in_channels = cfg.INPUT_CHANNELS 
        img_size = cfg.IMAGE_SIZE     
        flattened = False
        
        # [修改] 建立虛擬輸入 (Batch, Channel, H, W)
        dummy_input = torch.zeros(1, in_channels, img_size, img_size)
        
        tokens = self.arch_str.split('-')

        for token in tokens:
            if "C" in token and "FC" not in token: 
                out_channels, kernel_size = map(int, token.split('C'))
                self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
                self.layers.append(snn.Leaky(beta=cfg.BETA, threshold=cfg.THRESHOLD, spike_grad=self.spike_grad))
                
                with torch.no_grad():
                    dummy_input = nn.Conv2d(in_channels, out_channels, kernel_size)(dummy_input)
                
                in_channels = out_channels
                
            elif token == "PL":
                self.layers.append(nn.AvgPool2d(2))
                with torch.no_grad():
                    dummy_input = nn.AvgPool2d(2)(dummy_input)
                    
            elif "FC" in token:
                out_features = int(token.replace("FC", ""))
                if not flattened:
                    flat_dim = dummy_input.numel() 
                    self.layers.append(nn.Flatten(start_dim=1)) 
                    self.layers.append(nn.Linear(flat_dim, out_features))
                    flattened = True
                else:
                    prev_out = self.layers[-2].out_features
                    self.layers.append(nn.Linear(prev_out, out_features))
                
                self.layers.append(snn.Leaky(beta=cfg.BETA, threshold=cfg.THRESHOLD, spike_grad=self.spike_grad))

    def forward(self, x):
        mem_dict = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, snn.Leaky):
                mem_dict[i] = layer.init_leaky()

        spk_rec = []
        for step in range(x.size(0)):
            current_input = x[step]
            for i, layer in enumerate(self.layers):
                if isinstance(layer, snn.Leaky):
                    current_input, mem_dict[i] = layer(current_input, mem_dict[i])
                else:
                    current_input = layer(current_input)
            spk_rec.append(current_input)

        return torch.stack(spk_rec, dim=0)

# --- CSV 讀寫功能 (維持原樣，不需要更動) ---
def load_weights_from_csv(model, filename):
    print(f"正在從 {filename} 讀取權重...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            state_dict = model.state_dict()
            current_layer_name = None
            current_shape = None
            data_buffer = []

            for row in reader:
                if not row: continue 
                if row[0].startswith("# Layer:"):
                    if current_layer_name and data_buffer:
                        try:
                            np_array = np.array(data_buffer, dtype=np.float32)
                            tensor = torch.from_numpy(np_array.reshape(current_shape))
                            if current_layer_name in state_dict:
                                if state_dict[current_layer_name].shape == tensor.shape:
                                    state_dict[current_layer_name].copy_(tensor)
                        except Exception as e:
                            print(f"⚠️ 讀取層 {current_layer_name} 錯誤: {e}")
                    
                    part1 = row[0].split(": ")[1]
                    current_layer_name = part1
                    part2 = row[1].split(": ")[1]
                    current_shape = ast.literal_eval(part2)
                    data_buffer = []
                elif row[0].startswith("#"): continue
                else:
                    try: data_buffer.append([float(x) for x in row])
                    except: continue
            
            if current_layer_name and data_buffer:
                try:
                    np_array = np.array(data_buffer, dtype=np.float32)
                    tensor = torch.from_numpy(np_array.reshape(current_shape))
                    if current_layer_name in state_dict:
                        if state_dict[current_layer_name].shape == tensor.shape:
                            state_dict[current_layer_name].copy_(tensor)
                except Exception: pass
                    
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return False

def export_weights_to_csv(model, filename):
    print(f"正在將權重匯出至 {filename} ...")
    state_dict = model.state_dict()
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"# Model Architecture: {cfg.MODEL_ARCH}"])
        writer.writerow([])
        for key, tensor in state_dict.items():
            shape = tuple(tensor.shape)
            np_data = tensor.detach().cpu().numpy()
            writer.writerow([f"# Layer: {key}", f"Original Shape: {shape}"])
            if len(shape) == 4:
                flatten_dim = int(np.prod(shape[1:]))
                saved_data = np_data.reshape(shape[0], flatten_dim)
                writer.writerow([f"# Format: Flattened Filter"])
            elif len(shape) == 1:
                saved_data = np_data.reshape(1, -1)
                writer.writerow(["# Format: Bias vector"])
            elif len(shape) == 0:
                saved_data = np_data.reshape(1, 1)
                writer.writerow(["# Format: Scalar"])
            else:
                saved_data = np_data
                writer.writerow(["# Format: Matrix"])
            writer.writerows(saved_data.tolist())
            writer.writerow([]) 
    print("✅ 匯出完成！")