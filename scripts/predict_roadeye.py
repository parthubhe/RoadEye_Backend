from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO(
        r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights\best.pt"
    )

    model.predict(
        source=r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\assets\sample_image.png",
        imgsz=640,
        conf=0.4,
        iou=0.5,
        device=0,
        save=True,
        save_txt=False,
        save_conf=True,
        show=False
    )

if __name__ == "__main__":
    freeze_support()   # REQUIRED on Windows
    main()

# model.predict(
#     source=0,   # webcam
#     imgsz=640,
#     conf=0.3
# )
# model.export(format="onnx", opset=12)
# model.export(format="tflite", int8=True)
