# YOLOv11-COCO128-Evaluation
YOLOv11 model training, inference and evaluation on COCO128 dataset using Google Colab
# 🧠 YOLOv11 Object Detection - Evaluation on COCO128

This project demonstrates the training, inference, and evaluation of a YOLOv11 object detection model using the `coco128` dataset on Google Colab. The project aims to explore model performance through Mean Average Precision (mAP), precision, recall, and visual output of detections.

---

## 📁 Project Structure

/content/ │ ├── coco128/ # Dataset (images and labels) │ ├── images/train2017/ # Training images │ └── labels/train2017/ # Corresponding labels │ ├── runs/detect/ # Model outputs (inference results) │ ├── val5/ # Evaluation run results │ └── predict/ # Inference results on test image │ └── yolov11_project.ipynb # Main notebook with code and evaluation


---

## 📦 Installation

Before starting, install the required packages:

bash
pip install ultralytics

## 🚀 Model Training & Inference

Dataset Setup
The dataset used is the COCO128 sample provided by Ultralytics, already in YOLO format.
Model Used
YOLOv11 (based on YOLOv8 architecture via Ultralytics).
Inference Command
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Pretrained YOLOv8n model
model.train(data="coco128.yaml", epochs=3)  # Example training
Model Evaluation
metrics = model.val()

📊 Evaluation Metrics


Metric	Value
mAP@0.5	0.0069
mAP@0.5:0.95	0.0016
Precision	0.0106
Recall	0.0049
F1 Score (Mean)	0.0058


📝 Conclusion

The YOLOv11 model requires additional training for improved accuracy.
Consider using a larger dataset or training for more epochs.
Fine-tuning and experimentation with model size and hyperparameters is recommended.

📌 Author

Prathamesh lad
Student Project – YOLOv11 Evaluation Task
