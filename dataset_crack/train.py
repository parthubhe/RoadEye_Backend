import ultralytics.nn.tasks as tasks
from simam import SimAM
tasks.SimAM = SimAM

from ultralytics import YOLO

def main():
    model = YOLO("yolo26n-simam.yaml").load("yolo26n.pt")

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
        workers=2,
        project="runs/pipe_proto",
        name="yolo26n_simam_v1",
    )

if __name__ == "__main__":
    main()