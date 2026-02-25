import os
import cv2
import albumentations as A
import shutil
import random
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
SRC_ROOT = r"Datasets\RoadEyeUnified"
DST_ROOT = r"Datasets\RoadEyeUnifiedAug"

SPLITS = ["train", "val"]   # never augment test
AUGS_PER_IMAGE = 3
IMG_EXTS = [".jpg", ".jpeg", ".png"]

random.seed(42)

# -------------------------------
# AUGMENTATION FACTORY
# -------------------------------
def build_augmenter():
    return A.Compose(
        [
            # -----------------------
            # Lighting (day / night)
            # -----------------------
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=random.uniform(0.2, 0.45),
                    contrast_limit=random.uniform(0.2, 0.45)
                ),
                A.RandomGamma(
                    gamma_limit=(
                        random.randint(60, 90),    # darker
                        random.randint(110, 160)  # brighter
                    )
                ),
            ], p=0.7),

            # -----------------------
            # Blur (motion / focus)
            # -----------------------
            A.OneOf([
                A.MotionBlur(
                    blur_limit=random.choice([7, 9, 11, 13, 15])
                ),
                A.GaussianBlur(
                    blur_limit=random.choice([3, 5, 7])
                ),
            ], p=0.5),

            # -----------------------
            # Weather
            # -----------------------
            A.OneOf([
                A.RandomRain(
                    brightness_coefficient=random.uniform(0.7, 0.9),
                    rain_type="heavy"
                ),
                A.RandomFog(
                    fog_coef_range=(0.1, 0.35)
                ),
            ], p=0.3),

            # -----------------------
            # Shadows
            # -----------------------
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.3
            ),

            # -----------------------
            # Perspective (bumps)
            # -----------------------
            A.Perspective(
                scale=(0.03, 0.08),
                keep_size=True,
                p=0.3
            ),

            # -----------------------
            # Sensor noise
            # -----------------------
            A.GaussNoise(
                std_range=(0.01, 0.05),  # MUST be 0–1
                p=0.4
            ),

            # -----------------------
            # Mobile compression
            # -----------------------
            A.ImageCompression(
                quality_range=(35, 75),
                p=0.5
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
            clip=True
        )
    )

# -------------------------------
# LABEL HELPERS
# -------------------------------
def load_labels(label_path):
    boxes, classes = [], []
    with open(label_path) as f:
        for line in f:
            c, x, y, w, h = map(float, line.strip().split())
            boxes.append([x, y, w, h])
            classes.append(int(c))
    return boxes, classes

def save_labels(path, boxes, classes):
    with open(path, "w") as f:
        for c, (x, y, w, h) in zip(classes, boxes):
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# -------------------------------
# MAIN PIPELINE
# -------------------------------
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

        for i in range(AUGS_PER_IMAGE):
            augmenter = build_augmenter()

            try:
                augmented = augmenter(
                    image=image,
                    bboxes=boxes,
                    class_labels=classes
                )
            except Exception:
                continue

            if len(augmented["bboxes"]) == 0:
                continue

            out_img = f"{base}_aug{i}.jpg"
            out_lbl = f"{base}_aug{i}.txt"

            cv2.imwrite(
                os.path.join(img_dst, out_img),
                cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            )
            save_labels(
                os.path.join(lbl_dst, out_lbl),
                augmented["bboxes"],
                augmented["class_labels"]
            )

print("✅ Augmented dataset created at:", DST_ROOT)
