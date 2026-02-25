import os
import cv2
import albumentations as A
import shutil
import random
from tqdm import tqdm

SRC_ROOT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified_SingleClass"
DST_ROOT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified_SingleClass_Aug"

SPLITS = ["train", "val"]   # NEVER augment test
AUGS_PER_IMAGE = 1          # will bring ~12k → ~18k

IMG_EXTS = [".jpg", ".jpeg", ".png"]
random.seed(42)

def build_augmenter():
    return A.Compose(
        [
            # 1️⃣ Gamma
            A.RandomGamma(gamma_limit=(70, 150), p=0.7),

            # 2️⃣ Blur
            A.OneOf([
                A.MotionBlur(blur_limit=7),
                A.GaussianBlur(blur_limit=5)
            ], p=0.4),

            # 3️⃣ Squish / Stretch
            A.Affine(
                scale=(0.8, 1.2),
                keep_ratio=False,
                p=0.5
            ),

            # 4️⃣ Small rotation
            A.Rotate(
                limit=8,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.4
            )
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
            clip=True
        )
    )

def load_labels(path):
    boxes, classes = [], []
    with open(path) as f:
        for line in f:
            c, x, y, w, h = map(float, line.strip().split())
            boxes.append([x, y, w, h])
            classes.append(int(c))
    return boxes, classes

def save_labels(path, boxes, classes):
    with open(path, "w") as f:
        for c, (x, y, w, h) in zip(classes, boxes):
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

for split in SPLITS:

    img_src = os.path.join(SRC_ROOT, "images", split)
    lbl_src = os.path.join(SRC_ROOT, "labels", split)

    img_dst = os.path.join(DST_ROOT, "images", split)
    lbl_dst = os.path.join(DST_ROOT, "labels", split)

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    # Copy originals
    for f in os.listdir(img_src):
        shutil.copy(os.path.join(img_src, f), img_dst)
    for f in os.listdir(lbl_src):
        shutil.copy(os.path.join(lbl_src, f), lbl_dst)

    # Augment
    for img_name in tqdm(os.listdir(img_src), desc=f"Augmenting {split}"):

        if not any(img_name.lower().endswith(ext) for ext in IMG_EXTS):
            continue

        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_src, img_name)
        lbl_path = os.path.join(lbl_src, base + ".txt")

        if not os.path.exists(lbl_path):
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, classes = load_labels(lbl_path)

        if len(boxes) == 0:
            continue

        augmenter = build_augmenter()

        augmented = augmenter(
            image=image,
            bboxes=boxes,
            class_labels=classes
        )

        if len(augmented["bboxes"]) == 0:
            continue

        out_img = f"{base}_aug.jpg"
        out_lbl = f"{base}_aug.txt"

        cv2.imwrite(
            os.path.join(img_dst, out_img),
            cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
        )

        save_labels(
            os.path.join(lbl_dst, out_lbl),
            augmented["bboxes"],
            augmented["class_labels"]
        )

print("✅ Augmented dataset created.")
