# Copilot / AI Agent 使用說明 (專案專屬)

目的：協助 AI 編碼代理快速理解此專案的架構、重要檔案、開發工作流與專案慣例，以便能安全且有效地修改或新增功能。

重點速覽
- 專案類型：桌面 GUI 的 Spiking Neural Network (SNN) 訓練與測試平台，主要以 `tkinter` GUI 驅動訓練/辨識流程。
- 主要技術：PyTorch、snntorch（`spikegen`, `functional`）、torchvision、Pillow。
- 互動入口：執行 [gui_main.py](gui_main.py) 開啟 GUI。

大架構與資料流（為何這樣設計）
- 使用者在 GUI ([gui_main.py](gui_main.py)) 操作：設定參數 → 載入模型 → 手畫或匯入圖片 → 產生 spike train (`spikegen.rate`) → 將 spike 輸入模型 `snn_model.DynamicSCNN()` → 取得 `spk_rec`（time x batch x classes 類似張量）→ 計數後取 argmax 作預測。
- 訓練流程由 GUI 啟動（`run_training`），使用 torchvision datasets（MNIST / CIFAR10），DataLoader -> spikegen -> loss = `SF.ce_rate_loss()` -> backprop。
- 權重儲存：採 CSV 格式（非 .pt），命名策略為 `WEIGHTS_FILE = f"{DATASET_NAME}_{MODEL_ARCH}.csv"`，匯入/匯出由 `snn_model` 提供的 `load_weights_from_csv` / `export_weights_to_csv` 實現。

關鍵檔案（直接參考）
- [gui_main.py](gui_main.py) : GUI 與大部份工程流程（參數設定、載入模型、單筆修正訓練、整體訓練回圈、影像前處理）。
- [snn_config.py](snn_config.py) : 全域設定（`DEVICE`, `IMAGE_SIZE`, `INPUT_CHANNELS`, `TIME_STEPS`, `BATCH_SIZE`, `LEARNING_RATE`, `DATA_PATH`, `WEIGHTS_FILE` 等）。AI 修改「行為」時多看看這裡。
- [snn_model.py](snn_model.py) : 模型定義 `DynamicSCNN()` 與 CSV 權重匯入/匯出實作（確保接口相容）。
- [setting/install_env.py](setting/install_env.py) 與 [setting/cuda.py](setting/cuda.py) : 與環境設定、CUDA 檢查／啟用相關。

專案慣例與重要細節（AI 代理務必注意）
- 權重採 CSV 交換：任何修改 `DynamicSCNN` 層名稱或權重序列時，必須同步更新 `snn_model.export_weights_to_csv` 與 `load_weights_from_csv`，否則無法互通現有權重檔。
- GUI 會在 `apply_config()` 更新 `snn_config` 的變數，因此當新增 config 參數或改名時，請同步更新 `gui_main.py` 的 `apply_config` 與 `PRESETS`。
- 圖片流程：GUI 的輸入來源有兩種——畫布手寫 (`self.draw_image`) 或 `import_image()` 匯入；最終都會被 resize 與轉成 `L` 或 `RGB` 再經 `transforms.ToTensor()` 與 Normalize。若改變 `IMAGE_SIZE` 或 channel，務必同時更新 GUI 顯示標題與 preview 尺寸。
- spike 計算：輸入張量經 `spikegen.rate(input, num_steps=TIME_STEPS)`，模型輸出 `spk_rec`（時間維度在最前面）；訓練與預測皆以 `spk_rec.sum(dim=0)` 或 `sum(dim=0)` 取得類別脈衝計數。

開發 / 執行工作流（可直接執行的指令）
- 建議先安裝需求（參考 `setting/install_env.py`）：

```bash
python -m pip install torch torchvision snntorch pillow numpy
```

- 執行 GUI（主要開發入口）：

```bash
python gui_main.py
```

- 若要調整 CUDA 或 device，檢查/修改 [setting/cuda.py](setting/cuda.py) 與 [snn_config.py](snn_config.py) 的 `DEVICE` 定義。

AI 代理修改建議範例（具體可操作步驟）
- 若要新增一個新的模型層或更改架構字串映射：
  1. 更新 [snn_model.py](snn_model.py) 裡的 `DynamicSCNN` 定義。
  2. 更新 `snn_model.export_weights_to_csv` / `load_weights_from_csv` 以保持 CSV 格式連續性。
  3. 更新 [gui_main.py](gui_main.py) 的 `PRESETS`（若新增預設架構），並確保 `cfg.MODEL_ARCH` 對應新的名稱。

- 若要更改前處理（例如改用不同 Normalize）：直接修改 [gui_main.py](gui_main.py) 中 `_get_input_tensor()` 與 `run_training()` 內建立 `transform` 的地方；注意要同步 train/test 的 normalize 常數。

常見陷阱（避免浪費時間）
- 不要只改 `gui_main.py` 中的參數顯示文字，卻忘了改 `snn_config.py` 的實際變數，因為 `apply_config()` 會把 Entry 的值寫回 `snn_config`。
- 權重格式非 .pt：若把儲存改為 PyTorch native（`.pt`），需同時更新所有讀寫與權重命名流程。

當前代碼中可直接引用的具體 API / 例子
- 產生 spike train：`spikegen.rate(input_tensor, num_steps=cfg.TIME_STEPS)`（見 [gui_main.py](gui_main.py)）。
- 損失函數：`loss_fn = SF.ce_rate_loss()`（見 `train_single_sample` 與 `run_training`）。
- 權重命名範例：`cfg.WEIGHTS_FILE = f"{cfg.DATASET_NAME}_{cfg.MODEL_ARCH}.csv"`（見 `apply_config`）。

回饋與迭代
- 我已新增本檔案，請告訴我是否要：
  - 將權重儲存改成 `.pt`（需我檢查並修改 `snn_model.py`）；
  - 加入 CI / 測試指令（目前 repo 未包含測試）；
  - 或補上具體的 `requirements.txt`。 

---
（自動產生說明：本檔案由 AI 代理建立，請檢閱內容與檔案連結是否正確）
