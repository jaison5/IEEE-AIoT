import os
import torch
import multiprocessing
from ultralytics import YOLO
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
assert torch.cuda.is_available(), "❌ 找不到 GPU，請確認 CUDA 與 PyTorch 安裝正確"
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True  

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO('models/11n-3.pt')   

    results = model.train(
        data="yaml/20250105.yaml",  
        imgsz=1024,                 
        epochs=300,                 
        batch=12,                    
        workers=8,                  
        device=0,
        amp=True,                   
        val=True,
        pretrained=True,           
        save_period=10,
        project='gamechess',
        name='exp',
        lr0=0.005,
        lrf=0.1,
        patience=50,                
        warmup_epochs=5,           
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
