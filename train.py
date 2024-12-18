from ultralytics import YOLO

# 加载模型
# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# 训练模型
train_results = model.train(
    data="/media/hp/DATADRIVE1/gmr/目标检测/dataset/data.yaml",  # 数据配置文件的路径
    epochs=500,  # 训练的轮数
    batch = 32,
    imgsz=(960,544),  # 训练图像大小
    device="0,1",  # 运行的设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
)
