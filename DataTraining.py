from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='E:\\PycharmProjects\\YOLO model\\dataset.yaml', epochs=10)
