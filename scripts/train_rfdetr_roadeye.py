import torch
from rfdetr import RFDETRMedium

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATASET_DIR = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified_RFDETR"

NUM_CLASSES = 2

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    print("CUDA Available:", torch.cuda.is_available())

    model = RFDETRMedium(
        num_classes=NUM_CLASSES,
        pretrained=True
    )

    model.train(
        dataset_dir=DATASET_DIR,   # <-- THIS IS REQUIRED

        epochs=120,
        batch_size=6,
        lr=1e-4,
        weight_decay=1e-4,
        warmup_epochs=5,
        image_size=576,
        num_workers=8,
        device="cuda",
        amp=True
    )

    model.save("rfdetr_roadeye_medium.pth")
    print("Training complete.")


if __name__ == "__main__":
    main()
