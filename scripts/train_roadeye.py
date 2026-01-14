from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data=r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified\roadeye.yaml",
        epochs=100,
        imgsz=640,
        
        batch=16,          # RTX 4070 sweet spot
        device=0,
        workers=8,         # multiprocessing workers
        cache=False,       # RAM warning earlier, keep False

        optimizer="AdamW",
        lr0=0.002,
        cos_lr=True,
        weight_decay=0.0005,

        # -------- VALID AUGMENTATIONS --------
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        mosaic=0.5,
        mixup=0,

        project="runs/RoadEye",
        name="yolo11_multiclass_rtx4070",
        exist_ok=True,

        patience=20,
        amp=False
    )

if __name__ == "__main__":
    freeze_support()   # required on Windows
    main()
