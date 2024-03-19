from ultralytics import YOLO


if __name__=="__main__":
    # 加载模型
    model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

    # 使用模型
    model.train(data="data/txt/data.yaml", epochs=100, batch=10, device=0)  # 训练模型

    metrics = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
    success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

