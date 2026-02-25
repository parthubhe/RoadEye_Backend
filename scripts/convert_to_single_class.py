import os
import shutil

INPUT_DIR = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified"
OUTPUT_DIR = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified_SingleClass"

SPLITS = ["train", "val", "test"]

def convert_labels(src_lbl, dst_lbl):
    with open(src_lbl, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        # Force class = 0 (single pothole class)
        new_line = "0 " + " ".join(parts[1:])
        new_lines.append(new_line)

    with open(dst_lbl, "w") as f:
        f.write("\n".join(new_lines))

for split in SPLITS:
    print(f"Processing {split}...")

    img_src = os.path.join(INPUT_DIR, "images", split)
    lbl_src = os.path.join(INPUT_DIR, "labels", split)

    img_dst = os.path.join(OUTPUT_DIR, "images", split)
    lbl_dst = os.path.join(OUTPUT_DIR, "labels", split)

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    for img_name in os.listdir(img_src):
        shutil.copy(os.path.join(img_src, img_name),
                    os.path.join(img_dst, img_name))

    for lbl_name in os.listdir(lbl_src):
        convert_labels(
            os.path.join(lbl_src, lbl_name),
            os.path.join(lbl_dst, lbl_name)
        )

print("✅ Single-class dataset created.")
