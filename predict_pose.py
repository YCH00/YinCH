from ultralytics import YOLO

# ����ģ��
model = YOLO('yolov8n-pose.pt')  # ���عٷ�ģ��
model = YOLO('path/to/best.pt')  # �����Զ���ģ��

# ��ģ�ͽ���Ԥ��
results = model('https://ultralytics.com/images/bus.jpg')  # ��һ��ͼƬ��Ԥ��

