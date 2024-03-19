from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n-pose.pt')  # 加载官方模型
model = YOLO('path/to/best.pt')  # 加载自定义模型

# 用模型进行预测
results = model('https://ultralytics.com/images/bus.jpg')  # 在一张图片上预测

