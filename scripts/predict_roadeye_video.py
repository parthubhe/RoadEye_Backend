from ultralytics import YOLO
from multiprocessing import freeze_support
import os


def main():
    # Path to trained model
    model_path = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights\best.pt"

    # Input video  
    video_path = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\assets\IndianPotholes_Dashcam_2.mp4"

    # Output directory
    output_dir = r"runs/RoadEye/predict_video"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Run inference
    model.predict(
        source=video_path,
        device=0,            # RTX 4070
        imgsz=640,
        conf=0.25,            # confidence threshold
        iou=0.5,
        stream=False,         # True only if you want generator output
        save=True,            # saves output video
        save_txt=False,
        save_conf=True,
        project=output_dir,
        name="roadeye_video_1",
        exist_ok=True,
        vid_stride=3          # increase (2–5) for faster inference
    )


if __name__ == "__main__":
    freeze_support()   # REQUIRED on Windows
    main()
