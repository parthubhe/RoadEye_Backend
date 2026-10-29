from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/pipe_proto/yolo26n_v1-4/weights/best.pt")

    results = model.predict(
        source="sewer_remapped/test/images",
        save=True,
        conf=0.25,
        imgsz=640,
        project="runs/pipe_proto",
        name="test_inference",
    )

    print(f"Ran inference on {len(results)} images")
    print(f"Annotated images with boxes saved to runs/pipe_proto/test_inference/")

if __name__ == "__main__":
    main()