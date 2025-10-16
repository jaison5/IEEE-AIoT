import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()


    model = YOLO('models/yolo11n.pt')
    results = model.train(
        data = "yaml/20250105.yaml",
        imgsz = 1024,
        epochs = 500,    #世代數
        patience = 50,  #等待世代數，無改善則提前結束
        batch = 50,     #批次大小
        project = 'yolov8n_object_1',
        name = 'new01'
    )