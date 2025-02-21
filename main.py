
# Load a model
model = YOLO("yolov8n.yaml")

# Train the model
train_results = model.train(
    data="coco8.yaml",
    epochs=3,
)

