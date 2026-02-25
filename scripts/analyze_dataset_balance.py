import os
from collections import defaultdict, Counter
import numpy as np

ROOT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified"

SPLITS = ["train", "val", "test"]

CLASS_NAMES = {
    0: "night_pothole",
    1: "pothole"
}

print("\n================ DATASET BALANCE ANALYSIS ================\n")

global_box_counter = Counter()
global_image_counter = Counter()
source_counter = Counter()

for split in SPLITS:
    print(f"\n----- SPLIT: {split.upper()} -----")

    img_dir = os.path.join(ROOT, "images", split)
    lbl_dir = os.path.join(ROOT, "labels", split)

    box_counter = Counter()
    image_counter = Counter()

    total_images = 0
    total_boxes = 0

    for file in os.listdir(lbl_dir):
        if not file.endswith(".txt"):
            continue

        total_images += 1
        source_prefix = file.split("_")[0]  # kg, mwpd, rf
        source_counter[source_prefix] += 1

        with open(os.path.join(lbl_dir, file), "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        classes_in_image = set()

        for line in lines:
            cls = int(line.split()[0])
            box_counter[cls] += 1
            global_box_counter[cls] += 1
            classes_in_image.add(cls)
            total_boxes += 1

        for cls in classes_in_image:
            image_counter[cls] += 1
            global_image_counter[cls] += 1

    print(f"Images: {total_images}")
    print(f"Total Boxes: {total_boxes}")

    print("\nBox Distribution:")
    for cls in sorted(box_counter):
        print(
            f"  {CLASS_NAMES[cls]}: "
            f"{box_counter[cls]} boxes "
            f"({box_counter[cls]/total_boxes*100:.2f}%)"
        )

    print("\nImage Distribution (images containing class):")
    for cls in sorted(image_counter):
        print(
            f"  {CLASS_NAMES[cls]}: "
            f"{image_counter[cls]} images "
            f"({image_counter[cls]/total_images*100:.2f}%)"
        )

print("\n================ GLOBAL STATS ================\n")

total_global_boxes = sum(global_box_counter.values())
total_global_images = sum(global_image_counter.values())

print("Global Box Distribution:")
for cls in sorted(global_box_counter):
    print(
        f"  {CLASS_NAMES[cls]}: "
        f"{global_box_counter[cls]} boxes "
        f"({global_box_counter[cls]/total_global_boxes*100:.2f}%)"
    )

print("\nGlobal Image Distribution:")
for cls in sorted(global_image_counter):
    print(
        f"  {CLASS_NAMES[cls]}: "
        f"{global_image_counter[cls]} images"
    )

print("\nSource Distribution:")
for src, count in source_counter.items():
    print(f"  {src}: {count} images")

# Imbalance ratio
if len(global_box_counter) == 2:
    cls0 = global_box_counter[0]
    cls1 = global_box_counter[1]
    ratio = max(cls0, cls1) / min(cls0, cls1)
    print(f"\n⚠ Class imbalance ratio (max/min): {ratio:.2f}x")

print("\n=================================================\n")
