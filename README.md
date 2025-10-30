# A Modular AIoT Framework for Low-Latency Real-Time Robotic Teleoperation

A lightweight, high-performance YOLOv11-nano model for detecting GameChess pieces in AIoT applications.

---

## 🔗Contents
- [🤖 Overview](#overview)
- [🔗 System Module Integration](#system-module-integration)
- [✨ Features](#features)
- [📋 Specifications](#specifications)
- [🛠️ Hardware Requirements](#hardware-requirements)
- [🔧 Training Instructions](#training-instructions)
  - [Configure Dataset](#1-configure-dataset)
  - [Train YOLOv11-nano](#2-train-yolov11-nano)
  - [Evaluation](#3-evaluation)
- [💻 Usage](#usage)
  - [Inference on New Images](#inference-on-new-images)
  - [Edge Deployment](#edge-deployment)
- [📊 Performance Metrics](#performance-metrics)
- [🖼️ Detection Examples](#detection-examples)
- [🚀 Future Enhancements](#future-enhancements)
- [🤝 Contributing](#contributing)
- [🙏 Acknowledgements](#acknowledgements)
- [📝 License](#license)
## 🤖 Overview <a id="overview"></a>


This project features a complete YOLOv11-nano pipeline for chess piece or workshop items detection:

- Fast and efficient object detection suitable for edge devices
- Easy-to-use training and evaluation scripts
- Exportable to TorchScript or ONNX for embedded AIoT deployment
- Open-source dataset and scripts for reproducibility

---

### 🔗 System Module Integration <b id="system-module-integration"></b>

The AIoT teleoperation framework integrates multiple components for real-time operation:

- **Camera data** is acquired through [`02.webcamera.py`](https://github.com/jaison5/Jaison-AIot-max/blob/main/code/02.webcamera.py).  
- **ESP8266 Controller** (`ESP8266`): Handles I2C and Serial communications, receives MQTT messages, and drives the servo motors of the robotic arm. Supports message queueing with 32-byte transmission limit.  
  GitHub link: [ESP8266 code](https://github.com/jaison5/Jaison-AIot-max/blob/main/code/ESP8266)  
- **Flutter Hand Control** (`control code`): Provides user interface for arm manipulation, inverse kinematics (IK), claw control, and LiveKit video streaming. Includes incremental motion via long-press buttons and axis inversion options.  
  GitHub link: [Control code](https://github.com/jaison5/Jaison-AIot-max/blob/main/code/control%20code)  
- **I2C & Serial communication** are integrated with [`robot_armT2.ino`](https://github.com/jaison5/Jaison-AIot-max/blob/main/code/robot_armT2.ino).  
- **Training configuration** and prompts are defined in [`train2.py`](https://github.com/jaison5/Jaison-AIot-max/blob/main/code/train2.py).

This modular structure ensures smooth integration between vision, control, and communication layers within the AIoT ecosystem.


---

## ✨ Features <a id="features"></a>

- Lightweight YOLOv11-nano model
- Training on custom GameChess dataset
- High precision and recall for standard chess pieces
- Edge inference ready with TorchScript/ONNX export
- Open-source and fully documented

---

## 📋 Specifications <a id="specifications"></a>

| Parameter       | Value                |
|-----------------|----------------------|
| **Model** | YOLOv11-nano         |
| **Dataset** | Datasets            |
| **Input Size** | 1024×1024            |
| **Batch Size** | 12                   |
| **Epochs** | 300                  |
| **Framework** | PyTorch              |
| **Edge Deployment** | ESP8266   |

---

## 🛠️ Hardware Requirements <a id="hardware-requirements"></a>

- GPU recommended for training (CUDA-enabled)
- Standard CPU sufficient for inference
- Python 3.10+
- Required Python packages: `torch`, `ultralytics`, `opencv-python`, `numpy`

---

## 🔧 Training Instructions

Before starting the training process, please ensure that your environment is properly configured for **Python**, **PyTorch**, and (optionally) **CUDA** acceleration.

---

### 1️⃣ Install Python

Download and install **Python 3.10+** from the official website:  
👉 [https://www.python.org/downloads/](https://www.python.org/downloads/)

After installation, check your version:

```bash
python --version
```
Expected output:
```nginx
Python 3.10.x
```
### 2️⃣ Install CUDA (for NVIDIA GPU users)
If you are using an NVIDIA GPU, installing CUDA allows PyTorch to utilize GPU acceleration.
### 🔹 Step 1: Check your GPU model
```Bash
nvidia-smi
```
This displays your NVIDIA driver version and supported CUDA version.
### 🔹 Step 2: Download CUDA Toolkit
Go to the official NVIDIA site:
👉 [here](https://developer.nvidia.com/cuda-toolkit-archive)
Choose a version compatible with your driver (e.g., CUDA 12.1 or 12.4).
### 🔹 Step 3: Install CUDA (for NVIDIA GPUs, Windows)
If your system uses an NVIDIA GPU, you need to install CUDA and cuDNN to enable GPU acceleration.
### ✅ Recommended Installation Steps:
Go to the [NVIDIA CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads) Page.
Select:
Operating System: Windows
Architecture: x86_64
Version: Your Windows version (e.g., 11 or 10)
Installer Type: exe (local) (recommended)
After installation, verify CUDA is available:
```Bash
nvcc --version
```
### Install cuDNN (optional but recommended):
Visit [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (requires login)
Download the version matching your CUDA version (e.g., CUDA 12.1)
Extract and copy the `bin`, `include`, and `lib` folders into your CUDA installation directory (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\`).
To confirm PyTorch detects your GPU:
```python
import torch
print(torch.cuda.is_available())  # Should return True
```
### 🔹 Step 4: Configure Dataset
Update your `datasets` file with paths to training images and class names.
### 🔹 Step 5: Train the YOLOv11-nano Model [`train2.py`](https://github.com/jaison5/Jaison-AIot-max/blob/main/code/train2.py)
Create a Python file (e.g., `train_yolo.py`) and paste the following script:
```python
import os
import torch
import multiprocessing
from ultralytics import YOLO

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
assert torch.cuda.is_available(), "❌ Can't find the GPU. Please confirm that CUDA and PyTorch are installed correctly."
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True  

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO('models/11n-3.pt')   # Load model

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

    print("✅ The training is completed, and the results are stored in:", results.save_dir)
```
Run the script:
```bash
python train_yolo.py
```
### ✅ After completion:
Training results, weights, and logs will be stored in the `runs/train/exp` directory.

---

## 💻 Usage <a id="usage"></a>
Inference on New Images <a id="inference-on-new-images"></a>
create [webcamera.py]
```python
import cv2
from ultralytics import YOLO

model = YOLO('models/yolo11n.pt')
cap = cv2.VideoCpature(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        result = model(frame)

        annotated_frame = result[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
```
Run the script:
```bash
python webcamera.py
```
---

## 📊 Performance Metrics <a id="performance-metrics"></a>

| Class        | Images | Instances | Box Precision | Box Recall | mAP@50 | mAP@50-95 |
|-------------|--------|-----------|---------------|------------|--------|------------|
| Red King    | 16     | 16        | 0.992         | 0.904      | 0.992  | 0.637      |
| Gold King   | 33     | 33        | 1.000         | 1.000      | 0.947  | 0.730      |
| White Queen | 31     | 31        | 0.864         | 0.839      | 0.864  | 0.565      |
| Blue Queen  | 31     | 31        | 1.000         | 0.611      | 1.000  | 0.612      |
| White Rook  | 30     | 30        | 0.611         | 1.000      | 0.611  | 0.579      |
| Blue Rook   | 31     | 31        | 0.761         | 0.968      | 0.761  | 0.640      |
| White Knight| 25     | 25        | 0.995         | 1.000      | 0.995  | 0.667      |
| Blue Knight | 34     | 34        | 0.953         | 0.602      | 0.953  | 0.630      |
| White Bishop| 30     | 30        | 0.900         | 1.000      | 0.900  | 0.507      |
| Blue Bishop | 34     | 34        | 0.914         | 1.000      | 0.914  | 0.643      |
| White Pawn  | 32     | 32        | 1.000         | 0.881      | 1.000  | 0.701      |
| Blue Pawn   | 39     | 39        | 0.868         | 1.000      | 0.868  | 0.709      |

**Overall mAP:**  
- mAP@50: 0.979~0.995  
- mAP@50-95: 0.565~0.730  

**Speed per image:**  
- Preprocess: 0.2 ms  
- Inference: 10.2 ms  
- Loss: 0.0 ms  
- Postprocess: 1.1 ms


---

## 🖼️ Detection Examples <a id="detection-examples"></a>

| Input Image | Predicted Output |
|-------------|----------------|
| ![input1](https://github.com/jaison5/Robotic-AIoT-Max/blob/main/result/train_batch0.jpg)| ![output1](https://github.com/jaison5/Robotic-AIoT-Max/blob/main/result/val_batch0_pred.jpg)|
| ![input2](https://github.com/jaison5/Robotic-AIoT-Max/blob/main/result/train_batch1.jpg)| ![output2](https://github.com/jaison5/Robotic-AIoT-Max/blob/main/result/val_batch1_pred.jpg)|


---

## 🚀 Future Enhancements <a id="future-enhancements"></a>
Edge inference failover mechanisms

Formal usability and latency benchmarking

Integration with encrypted, authenticated communication channels

Multi-class AIoT perception for complex smart environments

ROS and mobile app integration

---

## 🤝 Contributing <a id="contributing"></a>
Collaboration is welcome! You can:

- Submit pull requests
- Report issues
- Suggest improvements

---

### 🙏 Acknowledgements <a id="acknowledgements"></a>

The support and contributions from the open-source community have been truly phenomenal.  
This project would not have been possible without the collective efforts of developers, researchers, and contributors who continuously push the boundaries of innovation.  
Your dedication and generosity embody the true spirit of open collaboration — thank you all for making this journey meaningful.

Special thanks to:

- [**Ultralytics**](https://github.com/ultralytics/ultralytics) — for providing the **YOLO framework**, which serves as the foundation for the object detection module in this project.  
- [**tzapu**](https://github.com/tzapu/WiFiManager?tab=readme-ov-file#contributions-and-thanks) — for developing and maintaining the **ESP8266 WiFiManager**, enabling seamless network configuration for embedded systems.  
- [**Gavinkuo123456**](https://github.com/Gavinkuo123456/CE2T-6-Axis-Robotic-Arm) — for sharing the **6-axis robotic arm design and control algorithms**, which inspired the robotic manipulation logic used here.  
- **AIoT research community** — for ongoing guidance, shared insights, and collaborative support throughout the development process.  
- **Open-source contributors** — for dataset management, training scripts, and continuous improvements that made this project possible.

This project builds upon the strong foundation laid by these open-source efforts.  
Your work and generosity make innovation possible.

**Thank you for your invaluable contributions!** 

---

## 📝 License <a id="license"></a>

This project is licensed under the MIT License - see the [LICENSE](https://github.com/jaison5/Robotic-AIoT-Max/blob/main/LICENSE) file for details.

---

## Thanks for your watching!
Shih-Chieh Sun, Yun-Cheng Tsai, “A Modular AIoT Framework for Low-Latency
Real-Time Robotic Teleoperation in Smart Cities,” IEEE AIoT, 2025.
