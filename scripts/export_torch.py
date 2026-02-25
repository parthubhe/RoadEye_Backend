from ultralytics import YOLO

# Load your trained model
model = YOLO(r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights\best.pt")

# Export to TorchScript (Required for this specific Android tutorial)
model.export(format="torchscript")