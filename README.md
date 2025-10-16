# A Modular AIoT Framework for Low-Latency Real-Time Robotic Teleoperation

A lightweight, high-performance YOLOv11-nano model for detecting GameChess pieces in AIoT applications.

---

## 🤖 Overview

This project features a complete YOLOv11-nano pipeline for chess piece or workshop items detection:

- Fast and efficient object detection suitable for edge devices
- Easy-to-use training and evaluation scripts
- Exportable to TorchScript or ONNX for embedded AIoT deployment
- Open-source dataset and scripts for reproducibility

---

### 🔗 System Module Integration

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

## ✨ Features

- Lightweight YOLOv11-nano model
- Training on custom GameChess dataset
- High precision and recall for standard chess pieces
- Edge inference ready with TorchScript/ONNX export
- Open-source and fully documented

---

## 📋 Specifications

| Parameter       | Value                |
|-----------------|----------------------|
| **Model** | YOLOv11-nano         |
| **Dataset** | Datasets            |
| **Input Size** | 1024×1024            |
| **Batch Size** | 12                   |
| **Epochs** | 300                  |
| **Framework** | PyTorch              |
| **Edge Deployment** | TorchScript / ONNX   |

---

## 🛠️ Hardware Requirements

- GPU recommended for training (CUDA-enabled)
- Standard CPU sufficient for inference
- Python 3.10+
- Required Python packages: `torch`, `ultralytics`, `opencv-python`, `numpy`

---

## 🔧 Training Instructions

### 1. Configure Dataset

Update `dataset.yaml` with paths to images and class names.

### 2. Train YOLOv11-nano

```bash
python train.py --model yolov11-nano.pt --data dataset.yaml --epochs 100 --batch-size 16 --img 1024
```

### 3. Evaluation

Evaluate the model performance after training:

```Bash
python val.py --weights runs/train/exp/weights/best.pt --data dataset.yaml --img 1024
```
---

## 💻 Usage
Inference on New Images

```Bash
python detect.py --weights runs/train/exp/weights/best.pt --source /path/to/images --img 1024
```
Edge Deployment

```Bash
python export.py --weights runs/train/exp/weights/best.pt --include torchscript onnx
```
---

## 📊 Performance Metrics

| Class  | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|-----------|--------|--------|------------|
| Pawn   | 0.95      | 0.92   | 0.94   | 0.88       |
| Rook   | 0.96      | 0.93   | 0.95   | 0.89       |
| Knight | 0.94      | 0.91   | 0.93   | 0.87       |
| Bishop | 0.95      | 0.92   | 0.94   | 0.88       |
| Queen  | 0.97      | 0.94   | 0.96   | 0.90       |

---

## 🖼️ Detection Examples

| Input Image | Predicted Output |
|-------------|----------------|
| ![input1](https://github.com/jaison5/IEEE-AIoT/blob/main/exp3/train_batch0.jpg) | ![output1](https://github.com/jaison5/IEEE-AIoT/blob/main/exp3/val_batch0_pred.jpg) |
| ![input2](https://github.com/jaison5/IEEE-AIoT/blob/main/exp3/train_batch1.jpg) | ![output2](https://github.com/jaison5/IEEE-AIoT/blob/main/exp3/val_batch1_pred.jpg) |


---

## 🚀 Future Enhancements
Edge inference failover mechanisms

Formal usability and latency benchmarking

Integration with encrypted, authenticated communication channels

Multi-class AIoT perception for complex smart environments

ROS and mobile app integration

---

## 🤝 Contributing
Contributions are welcome! You can:

Submit pull requests

Report issues

Suggest improvements

---

## 🙏 Acknowledgments
Ultralytics for the YOLO framework

AIoT research community for continuous support

Open-source contributors for dataset management and training scripts

---

## Thanks for your watching!
