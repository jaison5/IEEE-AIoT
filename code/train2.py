import os
import torch
import multiprocessing
from ultralytics import YOLO
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
assert torch.cuda.is_available(), "❌ 找不到 GPU，請確認 CUDA 與 PyTorch 安裝正確"
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True  # ✅ 自動選擇最佳卷積演算法

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO('models/11n-2.pt')   # 建議使用官方 YOLO11n 初始權重

    results = model.train(
        data="yaml/20250105.yaml",  # 資料集
        imgsz=1024,                 # ✅ 高解析度訓練，適合大顯存（1536~2048）
        epochs=300,                 # 訓練次數
        batch=12,                    # ✅ 建議 8~12；可依 VRAM 使用量調整
        workers=8,                  # 多執行緒載入
        device=0,
        amp=True,                   # 混合精度加速
        val=True,
        pretrained=True,            # 使用預訓練權重
        save_period=10,
        project='gamechess',
        name='exp',
        lr0=0.005,
        lrf=0.1,
        patience=50,                # 容忍度高一點
        warmup_epochs=5,            # 更穩定的初期收斂
        cos_lr=True,
        augment=True,
        mosaic=0.8,
        mixup=0.15,
        copy_paste=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.6,
        shear=2.0,
        perspective=0.0005
    )

    print("✅ 訓練完成，結果儲存於：", results.save_dir)
