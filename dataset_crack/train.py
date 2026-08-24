import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import CBAM
tasks.CBAM = CBAM

from ultralytics import YOLO

def main():
    model = YOLO("yolo26n-cbam.yaml").load("yolo26n.pt")

    print(type(model.model.model[17]).__name__)
    print(sum(p.numel() for p in model.model.parameters()))

    model.train(
        data="sewer_remapped/data.yaml",
        epochs=140,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.002,
        cos_lr=True,
        device=0,
        project="runs/pipe_proto",
        name="yolo26n_cbam_v1_140ep",
        workers=2,
    )

if __name__ == "__main__":
    main()