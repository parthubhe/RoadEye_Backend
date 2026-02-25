import os
import json
import shutil
import yaml
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_ROOT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified_SingleClass_Aug"
YAML_PATH = os.path.join(DATASET_ROOT, "roadeye.yaml")
SPLITS = ["train", "val"]

OUTPUT_ROOT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified_SingleClass_RFDETR_Aug"

# -------------------------------------------------
# LOAD CLASS NAMES FROM YAML (SAFE WAY)
# -------------------------------------------------
with open(YAML_PATH, "r") as f:
    data = yaml.safe_load(f)

# Ensures correct class order (0,1,...)
CLASSES = [data["names"][k] for k in sorted(data["names"].keys())]

print("Loaded classes:", CLASSES)

# -------------------------------------------------
# CONVERSION FUNCTION
# -------------------------------------------------
def convert_split(split):

    print(f"\nProcessing {split} split...")

    images_path = os.path.join(DATASET_ROOT, "images", split)
    labels_path = os.path.join(DATASET_ROOT, "labels", split)

    split_output_dir = os.path.join(OUTPUT_ROOT, split)
    images_output_dir = os.path.join(split_output_dir, "images")

    os.makedirs(images_output_dir, exist_ok=True)

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for idx, cls in enumerate(CLASSES):
        coco_output["categories"].append({
            "id": idx,
            "name": cls,
            "supercategory": "object"
        })


    annotation_id = 0
    image_id = 0

    image_files = os.listdir(images_path)

    for img_file in tqdm(image_files):

        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(images_path, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        # Copy image to RF-DETR folder
        shutil.copy2(img_path, os.path.join(images_output_dir, img_file))

        # Read image size
        img = Image.open(img_path)
        width, height = img.size

        coco_output["images"].append({
            "id": image_id,
            "file_name": f"images/{img_file}",
            "width": width,
            "height": height
        })

        # If label exists, convert
        if os.path.exists(label_path):

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                cls, x, y, w, h = map(float, line.strip().split())

                # YOLO → COCO conversion
                x_min = (x - w / 2) * width
                y_min = (y - h / 2) * height
                box_width = w * width
                box_height = h * height

                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(cls),
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

    # Save JSON
    output_json = os.path.join(split_output_dir, "annotations.json")

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"{split} conversion complete → {output_json}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for split in SPLITS:
        convert_split(split)

    print("\n✅ YOLO → COCO conversion finished successfully!")
