import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
import os
from torchvision import datasets, transforms
from snntorch import spikegen
import snntorch as snn

# 引用你原本的設定與模型，以及我們剛寫的 INT8 模組
import snn_config as cfg
import snn_model
import snn_int8  # <--- [新增] 匯入 INT8 模組

class SNNVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("SCNN 神經元激發視覺化工具 (含 INT8 模式)")
        self.root.geometry("1100x750")

        self.model = None
        self.device = cfg.DEVICE
        self.input_tensor = None
        self.image_display = None
        
        # 鉤子與特徵儲存
        self.hooks = []
        self.layer_outputs = {}

        self.setup_ui()
        self.on_dataset_change(None) 

    def setup_ui(self):
        # === 左側：參數與控制區 ===
        left_panel = ttk.LabelFrame(self.root, text="模型設定與控制")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 1. 資料庫選擇
        ttk.Label(left_panel, text="資料庫 (Dataset):").pack(anchor=tk.W, pady=(5,0))
        self.combo_dataset = ttk.Combobox(left_panel, values=list(cfg.PRESETS.keys()), state="readonly")
        self.combo_dataset.set("MNIST")
        self.combo_dataset.pack(fill=tk.X)
        self.combo_dataset.bind("<<ComboboxSelected>>", self.on_dataset_change)

        # 2. 參數設定
        def create_entry(label_text, var_name):
            ttk.Label(left_panel, text=label_text).pack(anchor=tk.W, pady=(5,0))
            entry = ttk.Entry(left_panel)
            entry.pack(fill=tk.X)
            setattr(self, f"entry_{var_name}", entry)

        create_entry("模型架構 (Model Arch):", "arch")
        create_entry("時間步長 (Time Steps):", "timesteps")
        create_entry("Beta (漏電率):", "beta")
        create_entry("Threshold (閾值):", "thresh")

        # 3. 權重與 INT8 設定
        weight_frame = ttk.LabelFrame(left_panel, text="權重與推論模式")
        weight_frame.pack(fill=tk.X, pady=10)
        
        self.var_custom_weight = tk.BooleanVar(value=False)
        ttk.Checkbutton(weight_frame, text="手動選擇權重", variable=self.var_custom_weight, command=self.toggle_weight).pack(anchor=tk.W)
        self.btn_browse = ttk.Button(weight_frame, text="瀏覽...", command=self.browse_weight, state=tk.DISABLED)
        self.btn_browse.pack(fill=tk.X, pady=2)
        self.lbl_weight = ttk.Label(weight_frame, text="(自動載入)", foreground="gray")
        self.lbl_weight.pack(fill=tk.X)
        self.custom_weight_path = ""
        
        # [新增] INT8 選擇器
        ttk.Separator(weight_frame, orient="horizontal").pack(fill=tk.X, pady=5)
        self.var_int8_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(weight_frame, text="⚡ 啟用 INT8 量化推論", variable=self.var_int8_mode).pack(anchor=tk.W, pady=(0, 5))

        # 4. 執行按鈕
        ttk.Button(left_panel, text="1. 隨機載入測試圖片", command=self.load_random_image).pack(fill=tk.X, pady=(20, 5))
        ttk.Button(left_panel, text="2. 開始推論並擷取特徵", command=self.run_inference).pack(fill=tk.X, pady=5)

        # 5. 狀態顯示
        self.lbl_status = ttk.Label(left_panel, text="狀態: 等待操作", foreground="blue")
        self.lbl_status.pack(pady=10)

        # === 右側：視覺化區 ===
        right_panel = ttk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 頂部：圖層選擇器
        selector_frame = ttk.Frame(right_panel)
        selector_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(selector_frame, text="檢視圖層:").pack(side=tk.LEFT, padx=(5, 2))
        self.combo_layer = ttk.Combobox(selector_frame, state="readonly", width=25)
        self.combo_layer.pack(side=tk.LEFT, padx=2)
        self.combo_layer.bind("<<ComboboxSelected>>", self.update_visualization)

        ttk.Label(selector_frame, text="時間步 (T):").pack(side=tk.LEFT, padx=(10, 2))
        self.combo_timestep = ttk.Combobox(selector_frame, state="readonly", width=12)
        self.combo_timestep.pack(side=tk.LEFT, padx=2)
        self.combo_timestep.bind("<<ComboboxSelected>>", self.update_visualization)

        ttk.Label(selector_frame, text="通道 (CH):").pack(side=tk.LEFT, padx=(10, 2))
        self.combo_channel = ttk.Combobox(selector_frame, state="readonly", width=8)
        self.combo_channel.pack(side=tk.LEFT, padx=2)
        self.combo_channel.bind("<<ComboboxSelected>>", self.update_visualization)

        # 底部：顯示區塊
        display_frame = ttk.Frame(right_panel)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 輸入圖片預覽
        input_frame = ttk.LabelFrame(display_frame, text="輸入圖片 (Float32)")
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.lbl_input_img = ttk.Label(input_frame, text="無圖片")
        self.lbl_input_img.pack(padx=20, pady=20)
        self.lbl_pred = ttk.Label(input_frame, text="預測結果: -", font=("Arial", 14, "bold"))
        self.lbl_pred.pack(pady=10)

        # 網格視覺化
        grid_frame = ttk.LabelFrame(display_frame, text="神經元脈衝激發方陣 (黑=發火, 白=安靜)")
        grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.canvas_size = 400
        self.canvas = tk.Canvas(grid_frame, width=self.canvas_size, height=self.canvas_size, bg="white", relief="sunken", borderwidth=2)
        self.canvas.pack(pady=20)
        self.lbl_grid_info = ttk.Label(grid_frame, text="等待數據...")
        self.lbl_grid_info.pack()

    # === 互動邏輯 ===
    def on_dataset_change(self, event):
        dataset = self.combo_dataset.get()
        settings = cfg.PRESETS.get(dataset, cfg.PRESETS["MNIST"])
        
        self.entry_arch.delete(0, tk.END)
        self.entry_arch.insert(0, settings["arch"])
        self.entry_timesteps.delete(0, tk.END)
        self.entry_timesteps.insert(0, str(settings["timesteps"]))
        self.entry_beta.delete(0, tk.END)
        self.entry_beta.insert(0, str(settings["beta"]))
        self.entry_thresh.delete(0, tk.END)
        self.entry_thresh.insert(0, str(settings["threshold"]))

    def toggle_weight(self):
        if self.var_custom_weight.get():
            self.btn_browse.config(state=tk.NORMAL)
        else:
            self.btn_browse.config(state=tk.DISABLED)
            self.lbl_weight.config(text="(自動載入)")

    def browse_weight(self):
        initial_dir = cfg.WEIGHT_DIR if os.path.exists(cfg.WEIGHT_DIR) else "."
        filename = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.custom_weight_path = filename
            self.lbl_weight.config(text=os.path.basename(filename), foreground="black")

    # === 核心運算 ===
    def apply_config(self):
        cfg.DATASET_NAME = self.combo_dataset.get()
        cfg.MODEL_ARCH = self.entry_arch.get()
        cfg.TIME_STEPS = int(self.entry_timesteps.get())
        cfg.BETA = float(self.entry_beta.get())
        cfg.THRESHOLD = float(self.entry_thresh.get())
        
        if cfg.DATASET_NAME == "MNIST":
            cfg.IMAGE_SIZE = 28
            cfg.INPUT_CHANNELS = 1
        else:
            cfg.IMAGE_SIZE = 32
            cfg.INPUT_CHANNELS = 3

        if self.var_custom_weight.get() and self.custom_weight_path:
            cfg.WEIGHTS_FILE = self.custom_weight_path
        else:
            cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHT_DIR, f"{cfg.DATASET_NAME}_{cfg.MODEL_ARCH}.csv")

    def load_random_image(self):
        self.apply_config()
        self.lbl_status.config(text="狀態: 下載/讀取資料集中...")
        self.root.update()

        try:
            if cfg.DATASET_NAME == "MNIST":
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                dataset = datasets.MNIST(root=cfg.DATA_PATH, train=False, download=True, transform=transform)
            else:
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
                dataset = datasets.CIFAR10(root=cfg.DATA_PATH, train=False, download=True, transform=transform)
            
            idx = np.random.randint(len(dataset))
            img_tensor, label = dataset[idx]
            
            self.input_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            img_np = img_tensor.permute(1, 2, 0).numpy()
            if cfg.INPUT_CHANNELS == 1:
                img_np = (img_np * 0.3081 + 0.1307) * 255
                img_np = img_np.squeeze()
                img_pil = Image.fromarray(img_np.astype(np.uint8), mode='L')
            else:
                img_np = (img_np * 0.5 + 0.5) * 255
                img_pil = Image.fromarray(img_np.astype(np.uint8), mode='RGB')
            
            img_display = img_pil.resize((150, 150), Image.Resampling.NEAREST)
            self.image_display = ImageTk.PhotoImage(img_display)
            self.lbl_input_img.config(image=self.image_display)
            
            current_labels = cfg.CLASS_LABELS.get(cfg.DATASET_NAME, cfg.CLASS_LABELS["MNIST"])
            self.lbl_pred.config(text=f"真實標籤: {current_labels[label]}")
            self.lbl_status.config(text="狀態: 圖片載入完成，請按開始推論")
            
            self.combo_layer.set('')
            self.combo_timestep.set('')
            self.combo_channel.set('')
            self.canvas.delete("all")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"讀取圖片失敗: {e}")

    def register_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []
        
        arch_tokens = cfg.MODEL_ARCH.split('-')
        leaky_tokens = [t for t in arch_tokens if t != "PL"]
        
        def hook_fn(module, input, output, name):
            spk = output[0].detach().cpu()[0] if isinstance(output, tuple) else output.detach().cpu()[0]
            if name not in self.layer_outputs:
                self.layer_outputs[name] = []
            self.layer_outputs[name].append(spk)

        leaky_idx = 0
        for i, layer in enumerate(self.model.layers):
            # [關鍵修改] 讓攔截器同時支援原本的 Float 神經元和新的 INT8 神經元
            if isinstance(layer, (snn.Leaky, snn_int8.INT8Leaky)):
                token_name = leaky_tokens[leaky_idx] if leaky_idx < len(leaky_tokens) else "Unknown"
                layer_name = f"{leaky_idx+1:02d}_{token_name}"
                
                h = layer.register_forward_hook(lambda m, i, o, n=layer_name: hook_fn(m, i, o, n))
                self.hooks.append(h)
                leaky_idx += 1

    def run_inference(self):
        if self.input_tensor is None:
            messagebox.showwarning("警告", "請先載入測試圖片！")
            return

        self.lbl_status.config(text="狀態: 載入模型與推論中...")
        self.root.update()

        try:
            self.apply_config()
            self.model = snn_model.DynamicSCNN().to(self.device)
            loaded = snn_model.load_weights_from_csv(self.model, cfg.WEIGHTS_FILE)
            if not loaded:
                messagebox.showwarning("警告", "找不到權重檔，目前使用隨機權重進行視覺化。")
            
            # [新增] 檢查是否啟用 INT8 模式
            if self.var_int8_mode.get():
                self.model = snn_int8.convert_to_int8(self.model)
                mode_str = "INT8"
            else:
                mode_str = "Float32"

            self.model.eval()
            self.layer_outputs = {}
            self.register_hooks()

            spike_data = spikegen.rate(self.input_tensor, num_steps=cfg.TIME_STEPS)
            self.layer_outputs["00_Input"] = [spike_data[t][0].cpu() for t in range(cfg.TIME_STEPS)]
            
            with torch.no_grad():
                spk_rec = self.model(spike_data)
                spike_counts = spk_rec.sum(dim=0).squeeze()
                pred_idx = spike_counts.argmax().item()
                
                current_labels = cfg.CLASS_LABELS.get(cfg.DATASET_NAME, cfg.CLASS_LABELS["MNIST"])
                current_text = self.lbl_pred.cget("text").split('\n')[0]
                
                # 預測結果顯示當前模式
                self.lbl_pred.config(text=f"{current_text}\n模型預測: {current_labels[pred_idx]} ({mode_str})")

            layer_names = list(self.layer_outputs.keys())
            self.combo_layer['values'] = layer_names
            if layer_names:
                self.combo_layer.current(0)
            
            time_options = ["All (Sum)"] + [f"T={t}" for t in range(cfg.TIME_STEPS)]
            self.combo_timestep['values'] = time_options
            self.combo_timestep.current(0)

            self.update_visualization()
            self.lbl_status.config(text=f"狀態: 推論完成 ({mode_str} 模式)！")
            
        except Exception as e:
            messagebox.showerror("推論錯誤", str(e))

    # === 動態繪圖邏輯 ===
    def update_visualization(self, event=None):
        layer_name = self.combo_layer.get()
        time_str = self.combo_timestep.get()
        channel_str = self.combo_channel.get()

        if not layer_name or layer_name not in self.layer_outputs: return
        if not time_str: return

        tensors = self.layer_outputs[layer_name] 
        
        if time_str == "All (Sum)":
            feature_map = torch.stack(tensors, dim=0).sum(dim=0)
        else:
            t_idx = int(time_str.replace("T=", ""))
            feature_map = tensors[t_idx]
            
        if feature_map.dim() == 1:
            feature_map = feature_map.unsqueeze(0) 

        num_channels = feature_map.shape[0]
        current_channels = self.combo_channel['values']
        
        if len(current_channels) != num_channels:
            self.combo_channel['values'] = [str(i) for i in range(num_channels)]
            self.combo_channel.current(0)
            channel_str = "0"
            
        channel_idx = int(channel_str) if channel_str else 0
        if channel_idx >= num_channels: channel_idx = 0
        
        matrix = feature_map[channel_idx].numpy()
        
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
            
        self.draw_grid(matrix)

    def draw_grid(self, matrix):
        self.canvas.delete("all")
        H, W = matrix.shape
        
        shape_type = "二維網格 (Conv/Input)" if H > 1 else "一維陣列 (FC)"
        self.lbl_grid_info.config(text=f"形狀: {W} x {H} | 類型: {shape_type}")

        cell_w = self.canvas_size / W
        
        if H == 1:
            cell_h = min(cell_w * 4, 100) 
            y_offset = (self.canvas_size - cell_h) / 2
        else:
            cell_h = self.canvas_size / H
            y_offset = 0

        outline_color = "#dddddd" if cell_w > 3 else ""

        for r in range(H):
            for c in range(W):
                val = matrix[r, c]
                fill_color = "black" if val > 0 else "white"
                
                x1 = c * cell_w
                y1 = y_offset + r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=outline_color)

if __name__ == "__main__":
    root = tk.Tk()
    app = SNNVisualizer(root)
    root.mainloop()