from ultralytics import YOLO

model = YOLO("./runs/segment/train5/weights/best.pt")  # load a pretrained YOLOv8n model
model.export(format="onnx",opset=12)  # export the model to ONNX forma