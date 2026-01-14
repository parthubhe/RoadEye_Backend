""" .pt to onnx conversion command:
yolo export model="D:\Study Material\Sem VI\Projects\PBL2\RoadEye\runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights\best.pt" format=onnx imgsz=640
"""
""".pt to tflite conversion command:
yolo export model="runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights\best.pt" format=tflite imgsz=640
"""
from ultralytics import YOLO

model = YOLO(
    r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights\best.pt"
)

model.export(
    format="tflite",
    imgsz=640
)
