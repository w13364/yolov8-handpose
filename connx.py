from ultralytics import YOLO
 
# Load a model
model = YOLO("best.pt")  # load an official model
 
# Export the model
model.export(format="rknn")