from ultralytics import YOLO

# 加载模型
# Load a model
# model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("/media/hp/DATADRIVE1/gmr/目标检测/runs/detect4090/train7/weights/best.pt")  # load a pretrained model (recommended for training)
results = model.val(data='/media/hp/DATADRIVE1/gmr/目标检测/dataset/data.yaml',device = '0')  # run YOLOv5x on COCO val with imgsz 640
print(results.box.map)