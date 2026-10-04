from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")
    model.train(
        data="final_dataset_v3/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.002,
        cos_lr=True,
        device=0,
        workers=6,
        project="runs/pipe_proto",
        name="yolo26n_final_merged",
    )

if __name__ == "__main__":
    main()