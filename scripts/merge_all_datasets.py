import os
import shutil
import random

random.seed(42)

OUT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified"

# --------- SOURCE DEFINITIONS ---------

SOURCES = [
    # Roboflow Night (only train)
    {
        "name": "rf",
        "pairs": [
            (
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\NightPotholesDataset\train\images",
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\NightPotholesDataset\train\labels"
            )
        ]
    },

    # MWPD Multiweather (train / val / test)
    {
        "name": "mwpd",
        "pairs": [
            (
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiweatherPotholeDataset\train\images",
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiweatherPotholeDataset\train\labels"
            ),
            (
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiweatherPotholeDataset\valid\images",
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiweatherPotholeDataset\valid\labels"
            ),
            (
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiweatherPotholeDataset\test\images",
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiweatherPotholeDataset\test\labels"
            )
        ]
    },

    # Kaggle Multiview (flat)
    {
        "name": "kg",
        "pairs": [
            (
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiviewPotholeDataset\images",
                r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiviewPotholeDataset\labels_yolo"
            )
        ]
    }
]

# --------- OUTPUT DIRS ---------

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

# --------- COLLECT ALL SAMPLES ---------

all_samples = []

for src in SOURCES:
    for img_dir, lbl_dir in src["pairs"]:
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        for img in os.listdir(img_dir):
            stem, ext = os.path.splitext(img)
            lbl = stem + ".txt"

            lbl_path = os.path.join(lbl_dir, lbl)
            img_path = os.path.join(img_dir, img)

            if os.path.exists(lbl_path):
                all_samples.append({
                    "src": src["name"],
                    "img": img_path,
                    "lbl": lbl_path,
                    "ext": ext
                })

print(f"📦 Total samples collected: {len(all_samples)}")

# --------- SHUFFLE & SPLIT ---------

random.shuffle(all_samples)
n = len(all_samples)

train_end = int(0.7 * n)
val_end   = int(0.9 * n)

splits = {
    "train": all_samples[:train_end],
    "val":   all_samples[train_end:val_end],
    "test":  all_samples[val_end:]
}

# --------- COPY WITH SAFE RENAME ---------

global_counter = 0

for split, items in splits.items():
    for sample in items:
        new_name = f"{sample['src']}_{global_counter:07d}"

        shutil.copy(
            sample["img"],
            os.path.join(OUT, "images", split, new_name + sample["ext"])
        )
        shutil.copy(
            sample["lbl"],
            os.path.join(OUT, "labels", split, new_name + ".txt")
        )

        global_counter += 1

print("✅ Unified dataset created successfully")
print(f"📂 Output path: {OUT}")
