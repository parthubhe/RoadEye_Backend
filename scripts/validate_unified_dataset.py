import os
from collections import Counter

# --------- UPDATE THIS TO YOUR UNIFIED DATASET PATH ---------
ROOT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified"

image_splits = {
    "train": os.path.join(ROOT, "images", "train"),
    "val":   os.path.join(ROOT, "images", "val"),
    "test":  os.path.join(ROOT, "images", "test")
}

label_splits = {
    "train": os.path.join(ROOT, "labels", "train"),
    "val":   os.path.join(ROOT, "labels", "val"),
    "test":  os.path.join(ROOT, "labels", "test")
}

print("\n========== Unified Dataset Validator ==========\n")

total_img = 0
total_lbl = 0

class_counter = Counter()
empty_label_files = []

for split in image_splits:
    img_dir = image_splits[split]
    lbl_dir = label_splits[split]

    print(f"→ Checking split: {split}")

    imgs = os.listdir(img_dir)
    for im in imgs:
        stem = os.path.splitext(im)[0]
        lbl_file = stem + ".txt"

        total_img += 1

        img_path = os.path.join(img_dir, im)
        lbl_path = os.path.join(lbl_dir, lbl_file)

        if not os.path.exists(lbl_path):
            print(f"❌ Missing label for {img_path}")
            continue

        total_lbl += 1

        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            empty_label_files.append(lbl_path)
            continue

        for ln in lines:
            parts = ln.split()
            cls = int(parts[0])
            class_counter[cls] += 1

    print(f"✔ {len(imgs)} images checked in {split}\n")

print("=========== Validation Summary ===========\n")
print(f"Total images seen : {total_img}")
print(f"Total labels seen : {total_lbl}")
print(f"Empty label files : {len(empty_label_files)}")

if empty_label_files:
    print("\nList of empty label files:")
    for p in empty_label_files[:10]:
        print("  ", p)
print("\nClass distribution:")
print("Class : 0 -> Night Time potholes, 1 -> Potholes")
for cls, cnt in class_counter.items():
    print(f"  class {cls} : {cnt} boxes")

print("\nValidation completed!\n")
