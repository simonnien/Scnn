import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw, ImageTk
import threading
import torch
import numpy as np
import sys
import os
from snntorch import spikegen, functional as SF
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snn_config as cfg
import snn_model

class SNN_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SCNN 通用圖像辨識平台 (Advanced Config)")
        self.root.geometry("1000x800") # 稍微拉高以容納新按鈕

        self.model = None
        self.device = cfg.DEVICE
        self.test_image_ref = None 
        self.is_training = False
        self.imported_image = None
        
        # 手動權重相關變數
        self.var_use_custom_weight = tk.BooleanVar(value=False)
        self.custom_weight_path = ""
        
        self.canvas_size = 200
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.draw_image)
        
        self.setup_ui()
        self.on_dataset_change(None, init=True)

    def setup_ui(self):
        # === 左側：參數設定區 ===
        left_panel = ttk.LabelFrame(self.root, text="參數設定 (Configuration)")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 1. 資料庫選擇
        dataset_frame = ttk.Frame(left_panel)
        dataset_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dataset_frame, text="資料庫 (Dataset):").pack(anchor=tk.W)
        
        dataset_options = list(cfg.PRESETS.keys())
        self.combo_dataset = ttk.Combobox(dataset_frame, values=dataset_options, width=28, state="readonly")
        self.combo_dataset.set("MNIST")
        self.combo_dataset.pack(fill=tk.X)
        self.combo_dataset.bind("<<ComboboxSelected>>", self.on_dataset_change)
        
        self.lbl_input_spec = ttk.Label(left_panel, text="Input: ???", foreground="blue")
        self.lbl_input_spec.pack(pady=2)

        # [新增] 權重檔案選擇區塊
        weight_frame = ttk.LabelFrame(left_panel, text="權重檔案 (Weights)")
        weight_frame.pack(fill=tk.X, pady=10)
        
        # 勾選框
        self.chk_custom = ttk.Checkbutton(weight_frame, text="手動選擇權重檔", 
                                          variable=self.var_use_custom_weight, 
                                          command=self.toggle_weight_selection)
        self.chk_custom.pack(anchor=tk.W, padx=5)
        
        # 選擇按鈕與顯示標籤
        self.btn_browse = ttk.Button(weight_frame, text="瀏覽 (Browse)...", 
                                     command=self.browse_weight_file, state=tk.DISABLED)
        self.btn_browse.pack(fill=tk.X, padx=5, pady=2)
        
        self.lbl_weight_path = ttk.Label(weight_frame, text="(使用自動命名)", foreground="gray", wraplength=200)
        self.lbl_weight_path.pack(fill=tk.X, padx=5, pady=2)

        # 一般參數輸入框
        def create_entry(label_text, var_name):
            frame = ttk.Frame(left_panel)
            frame.pack(fill=tk.X, pady=5)
            ttk.Label(frame, text=label_text).pack(anchor=tk.W)
            entry = ttk.Entry(frame, width=30)
            entry.pack(fill=tk.X)
            setattr(self, f"entry_{var_name}", entry)

        create_entry("模型架構 (Model Arch):", "arch")
        create_entry("訓練輪數 (Epochs):", "epochs")
        create_entry("時間步長 (Time Steps):", "timesteps")
        create_entry("Beta (漏電率):", "beta")
        create_entry("Threshold (閾值):", "thresh")
        create_entry("Batch Size:", "batch")
        create_entry("Learning Rate:", "lr")

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(pady=20, fill=tk.X)
        ttk.Button(btn_frame, text="套用參數並重載", command=self.apply_config).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="⚠ 重啟程式 (Reset App)", command=self.restart_program).pack(fill=tk.X, pady=10)

        # === 右側：功能區 (維持不變) ===
        right_panel = ttk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_train = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="訓練模式 (Train)")
        self.setup_train_tab()

        self.tab_test = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_test, text="測試模式 (Test / Correction)")
        self.setup_test_tab()

    # ... (setup_train_tab, setup_test_tab, on_dataset_change, restart_program 保持不變) ...
    def setup_train_tab(self):
        top_frame = ttk.Frame(self.tab_train)
        top_frame.pack(pady=10, fill=tk.X)
        self.btn_train = ttk.Button(top_frame, text="開始訓練 (Start)", command=self.start_training_thread)
        self.btn_train.pack(side=tk.LEFT, padx=10)
        self.btn_stop = ttk.Button(top_frame, text="強制中斷 (Stop)", command=self.stop_training, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        self.log_text = tk.Text(self.tab_train, height=25, width=60, state=tk.DISABLED)
        self.log_text.pack(pady=10, fill=tk.BOTH, expand=True)

    def setup_test_tab(self):
        draw_area = ttk.Frame(self.tab_test)
        draw_area.pack(pady=5)
        left_draw = ttk.Frame(draw_area)
        left_draw.pack(side=tk.LEFT, padx=20)
        ttk.Label(left_draw, text="手寫/圖片區").pack()
        self.canvas = tk.Canvas(left_draw, width=self.canvas_size, height=self.canvas_size, bg="black", cursor="cross")
        self.canvas.pack(pady=5)
        self.canvas.bind("<B1-Motion>", self.paint)
        right_preview = ttk.Frame(draw_area)
        right_preview.pack(side=tk.LEFT, padx=20)
        self.lbl_preview_title = ttk.Label(right_preview, text="模型視野")
        self.lbl_preview_title.pack()
        self.lbl_test_preview = ttk.Label(right_preview, text="無影像", relief="sunken")
        self.lbl_test_preview.pack(pady=5)
        empty_img = Image.new("RGB", (140, 140), "black")
        self.test_image_ref = ImageTk.PhotoImage(empty_img)
        self.lbl_test_preview.config(image=self.test_image_ref)
        btn_frame = ttk.Frame(self.tab_test)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="清除 (Clear)", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="匯入圖片 (Import)", command=self.import_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="辨識 (Predict)", command=self.predict_digit).pack(side=tk.LEFT, padx=5)
        train_single_frame = ttk.LabelFrame(self.tab_test, text="單筆糾錯訓練 (Correction)")
        train_single_frame.pack(pady=5, fill=tk.X, padx=20)
        ttk.Label(train_single_frame, text="正確類別:").pack(side=tk.LEFT, padx=5)
        self.combobox_label = ttk.Combobox(train_single_frame, width=10, state="readonly")
        self.combobox_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(train_single_frame, text="以此圖訓練並存檔", command=self.train_single_sample).pack(side=tk.LEFT, padx=10)
        self.lbl_result = ttk.Label(self.tab_test, text="預測結果: ???", font=("Arial", 16, "bold"))
        self.lbl_result.pack(pady=10)
        self.lbl_spikes = ttk.Label(self.tab_test, text="脈衝計數: []", font=("Courier", 10))
        self.lbl_spikes.pack()

    def on_dataset_change(self, event, init=False):
        dataset = self.combo_dataset.get()
        settings = cfg.PRESETS.get(dataset, cfg.PRESETS["MNIST"])
        current_labels = cfg.CLASS_LABELS.get(dataset, cfg.CLASS_LABELS["MNIST"])
        self.combobox_label['values'] = current_labels
        self.combobox_label.current(0) 
        def set_val(entry, value):
            entry.delete(0, tk.END)
            entry.insert(0, str(value))
        set_val(self.entry_arch, settings["arch"])
        set_val(self.entry_epochs, settings["epochs"])
        set_val(self.entry_timesteps, settings["timesteps"])
        set_val(self.entry_beta, settings["beta"])
        set_val(self.entry_thresh, settings["threshold"])
        set_val(self.entry_batch, settings["batch"])
        set_val(self.entry_lr, settings["lr"])
        self.apply_config(initial=init)

    def restart_program(self):
        if messagebox.askyesno("確認", "確定要重新啟動程式嗎？"):
            python = sys.executable
            os.execl(python, python, *sys.argv)

    # === [新增] 權重檔案選擇邏輯 ===

    def toggle_weight_selection(self):
        """切換手動/自動模式"""
        if self.var_use_custom_weight.get():
            # 啟用手動模式
            self.btn_browse.config(state=tk.NORMAL)
            if self.custom_weight_path:
                self.lbl_weight_path.config(text=os.path.basename(self.custom_weight_path), foreground="black")
            else:
                self.lbl_weight_path.config(text="請選擇檔案...", foreground="red")
        else:
            # 停用手動模式 (回歸自動)
            self.btn_browse.config(state=tk.DISABLED)
            self.lbl_weight_path.config(text="(使用自動命名)", foreground="gray")

    def browse_weight_file(self):
        """開啟檔案瀏覽器"""
        # 預設開啟 ./WEIGHT/ 資料夾
        initial_dir = cfg.WEIGHT_DIR if os.path.exists(cfg.WEIGHT_DIR) else "."
        filename = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="選擇權重檔案",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if filename:
            self.custom_weight_path = filename
            # 顯示簡短檔名
            self.lbl_weight_path.config(text=os.path.basename(filename), foreground="black")

    # === [修改] 應用設定 (Apply Config) ===
    
    def apply_config(self, initial=False):
        try:
            cfg.DATASET_NAME = self.combo_dataset.get()
            
            # 設定圖片規格
            if cfg.DATASET_NAME == "MNIST":
                cfg.IMAGE_SIZE = 28
                cfg.INPUT_CHANNELS = 1
            elif cfg.DATASET_NAME == "CIFAR10":
                cfg.IMAGE_SIZE = 32
                cfg.INPUT_CHANNELS = 3
            
            self.lbl_input_spec.config(text=f"Specs: {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}, {cfg.INPUT_CHANNELS} Chs")
            self.lbl_preview_title.config(text=f"模型視野 ({cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE})")

            # 讀取數值參數
            cfg.MODEL_ARCH = self.entry_arch.get()
            cfg.NUM_EPOCHS = int(self.entry_epochs.get())
            cfg.TIME_STEPS = int(self.entry_timesteps.get())
            cfg.BETA = float(self.entry_beta.get())
            cfg.THRESHOLD = float(self.entry_thresh.get())
            cfg.BATCH_SIZE = int(self.entry_batch.get())
            cfg.LEARNING_RATE = float(self.entry_lr.get())
            
            # [修改] 決定權重檔路徑
            if self.var_use_custom_weight.get() and self.custom_weight_path:
                # 使用手動選擇的路徑
                cfg.WEIGHTS_FILE = self.custom_weight_path
            else:
                # 使用自動命名 (包含 ./WEIGHT/ 路徑)
                filename = f"{cfg.DATASET_NAME}_{cfg.MODEL_ARCH}.csv"
                cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHT_DIR, filename)
            
            # 嘗試載入模型
            self.load_current_model()
            
            if not initial:
                # 提示訊息增加顯示目前的權重檔名
                msg = f"參數更新！\n資料庫: {cfg.DATASET_NAME}\n權重檔: {os.path.basename(cfg.WEIGHTS_FILE)}"
                messagebox.showinfo("成功", msg)
                
        except ValueError as e:
            if not initial:
                messagebox.showerror("參數錯誤", f"格式錯誤: {e}")

    # ... (其餘函式 load_current_model, paint, import_image, clear_canvas, 
    #      _get_input_tensor, predict_digit, train_single_sample, 
    #      log_message, stop_training, start_training_thread, run_training 
    #      皆保持不變) ...

    def load_current_model(self):
        try:
            self.model = snn_model.DynamicSCNN().to(self.device)
            loaded = snn_model.load_weights_from_csv(self.model, cfg.WEIGHTS_FILE)
            self.model.eval()
            msg = f"已載入模型: {os.path.basename(cfg.WEIGHTS_FILE)}" if loaded else "權重檔不存在，使用隨機初始化"
            self.log_message(msg)
        except Exception as e:
            self.log_message(f"模型建構失敗: {e}")

    def paint(self, event):
        if self.imported_image:
            self.imported_image = None
            self.log_message("偵測到手寫，已移除匯入圖片。")
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)

    def import_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            try:
                img = Image.open(file_path)
                self.imported_image = img.copy()
                img_display = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS)
                self.tk_imported_img = ImageTk.PhotoImage(img_display)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_imported_img)
                self.log_message(f"已匯入圖片: {os.path.basename(file_path)}")
                self.lbl_result.config(text="圖片已匯入，請點擊辨識")
            except Exception as e:
                messagebox.showerror("錯誤", f"無法開啟圖片: {e}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.draw_image)
        self.imported_image = None
        self.lbl_result.config(text="預測結果: ???")
        empty_img = Image.new("RGB", (140, 140), "black")
        self.test_image_ref = ImageTk.PhotoImage(empty_img)
        self.lbl_test_preview.config(image=self.test_image_ref)

    def _get_input_tensor(self):
        if self.imported_image:
            source_img = self.imported_image
        else:
            source_img = self.draw_image

        target_size = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        img_resized = source_img.resize(target_size, Image.Resampling.LANCZOS)
        
        if cfg.INPUT_CHANNELS == 1:
            img_final = img_resized.convert("L")
            norm_mean, norm_std = (0.5,), (0.5,)
        else:
            img_final = img_resized.convert("RGB")
            norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        display_img = img_final.resize((140, 140), Image.Resampling.NEAREST)
        self.test_image_ref = ImageTk.PhotoImage(display_img)
        self.lbl_test_preview.config(image=self.test_image_ref, text="")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
        return transform(img_final).unsqueeze(0).to(self.device)

    def predict_digit(self):
        if self.model is None: return
        input_tensor = self._get_input_tensor()
        spike_data = spikegen.rate(input_tensor, num_steps=cfg.TIME_STEPS)

        with torch.no_grad():
            self.model.eval()
            spk_rec = self.model(spike_data)
            spike_counts = spk_rec.sum(dim=0).squeeze()
            pred_idx = spike_counts.argmax().item()
            current_labels = cfg.CLASS_LABELS.get(cfg.DATASET_NAME, cfg.CLASS_LABELS["MNIST"])
            label_name = current_labels[pred_idx] if pred_idx < len(current_labels) else str(pred_idx)
            self.lbl_result.config(text=f"預測: {label_name} ({pred_idx})")
            self.lbl_spikes.config(text=f"計數: {spike_counts.cpu().numpy().astype(int).tolist()}")

    def train_single_sample(self):
        if self.model is None: return
        selected_text = self.combobox_label.get()
        current_labels = cfg.CLASS_LABELS.get(cfg.DATASET_NAME, cfg.CLASS_LABELS["MNIST"])
        try:
            target_idx = current_labels.index(selected_text)
        except ValueError:
            messagebox.showerror("錯誤", "無法識別選擇的類別")
            return
        target = torch.tensor([target_idx]).to(self.device)
        input_tensor = self._get_input_tensor()
        spike_data = spikegen.rate(input_tensor, num_steps=cfg.TIME_STEPS)
        try:
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.LEARNING_RATE)
            loss_fn = SF.ce_rate_loss()
            spk_rec = self.model(spike_data)
            loss = loss_fn(spk_rec, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            snn_model.export_weights_to_csv(self.model, cfg.WEIGHTS_FILE)
            self.model.eval()
            self.lbl_result.config(text=f"單筆訓練完成! Loss: {loss.item():.4f}")
            messagebox.showinfo("成功", f"圖片已學習！\n目標: {selected_text}\nSaved to: {cfg.WEIGHTS_FILE}")
        except Exception as e:
            messagebox.showerror("訓練錯誤", f"單筆訓練失敗: {e}")

    def log_message(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def stop_training(self):
        if self.is_training:
            self.is_training = False
            self.log_message("⚠ 正在中斷訓練...")
            self.btn_stop.config(state=tk.DISABLED)

    def start_training_thread(self):
        self.is_training = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        try:
            epochs = int(self.entry_epochs.get())
            self.log_message(f"--- 開始訓練 ({cfg.DATASET_NAME}, Epochs: {epochs}) ---")
            
            if cfg.DATASET_NAME == "MNIST":
                transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                train_dataset = datasets.MNIST(root=cfg.DATA_PATH, train=True, download=True, transform=transform)
            elif cfg.DATASET_NAME == "CIFAR10":
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                train_dataset = datasets.CIFAR10(root=cfg.DATA_PATH, train=True, download=True, transform=transform)
            
            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.LEARNING_RATE)
            loss_fn = SF.ce_rate_loss()
            
            self.model.train()
            
            for epoch in range(epochs):
                if not self.is_training: break
                total_loss = 0
                total_correct = 0
                total_samples = 0
                for i, (data, targets) in enumerate(train_loader):
                    if not self.is_training: break
                    data, targets = data.to(self.device), targets.to(self.device)
                    spike_data = spikegen.rate(data, num_steps=cfg.TIME_STEPS)
                    spk_rec = self.model(spike_data)
                    loss = loss_fn(spk_rec, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    spike_count = spk_rec.sum(dim=0)
                    _, predicted = spike_count.max(1)
                    correct = (predicted == targets).sum().item()
                    total_correct += correct
                    total_samples += targets.size(0)
                    if (i + 1) % 50 == 0:
                         batch_acc = 100 * correct / targets.size(0)
                         self.log_message(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%")

                if self.is_training:
                    avg_loss = total_loss / len(train_loader)
                    avg_acc = 100 * total_correct / total_samples
                    self.log_message(f"==> Epoch {epoch+1} 完成。Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
                else:
                    self.log_message(">> 訓練已手動中斷。")
            
            snn_model.export_weights_to_csv(self.model, cfg.WEIGHTS_FILE)
            self.log_message(f"權重已儲存至: {cfg.WEIGHTS_FILE}")
        except Exception as e:
            self.log_message(f"訓練錯誤: {e}")
        finally:
            self.is_training = False
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

if __name__ == "__main__":
    root = tk.Tk()
    app = SNN_GUI(root)
    root.mainloop()