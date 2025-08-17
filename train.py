from ultralytics import YOLO

# 加载模型
model = YOLO("./yolov8n-cls.pt")  # 从YAML构建并转移权重

if __name__ == "__main__":
    # Train the model
    results = model.train(data="./flowers2", epochs=10, imgsz=64)
