from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO(
        r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights\best.pt"
    )

    model.val(
        data=r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified\roadeye.yaml",
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        plots=True
    )

if __name__ == "__main__":
    freeze_support()   # REQUIRED on Windows
    main()
