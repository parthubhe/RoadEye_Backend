import cv2
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import os

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_PATH = r"D:\\Study Material\\Sem VI\\Projects\\PBL2\\RoadEye\\Datasets\\RoadEyeUnified\\images\\train\\kg_0000055.png"  # <-- put a pothole image here
SAVE_OUTPUT = False        # set True if you want saved images
OUT_DIR = "augmentation_demo"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# LOAD IMAGE
# -------------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------------------------------
# AUGMENTATION PIPELINE
# -------------------------------
augmentations = {
    "Original": A.Compose([]),

    "Brightness / Contrast": A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.35,
            contrast_limit=0.35,
            p=1.0
        )
    ]),

    "Gamma (Low Light)": A.Compose([
        A.RandomGamma(
            gamma_limit=(40, 120),
            p=1.0
        )
    ]),

    "Gaussian Blur": A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0)
    ]),

    "Motion Blur": A.Compose([
        A.MotionBlur(blur_limit=15, p=1.0)
    ]),

    "Rain": A.Compose([
        A.RandomRain(
            blur_value=3,
            brightness_coefficient=0.9,
            rain_type="heavy",
            p=1.0
        )
    ]),

    "Fog / Haze": A.Compose([
        A.RandomFog(
            fog_coef_lower=0.3,
            fog_coef_upper=0.6,
            alpha_coef=0.08,
            p=1.0
        )
    ]),

    "Shadow": A.Compose([
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=3,
            p=1.0
        )
    ]),

    "Perspective": A.Compose([
        A.Perspective(scale=(0.03, 0.08), p=1.0)
    ]),

    "Noise": A.Compose([
        A.GaussNoise(var_limit=(10, 50), p=1.0)
    ]),

    "JPEG Compression": A.Compose([
        A.ImageCompression(quality_lower=30, quality_upper=70, p=1.0)
    ])
}

# -------------------------------
# APPLY + VISUALIZE
# -------------------------------
plt.figure(figsize=(16, 12))

for idx, (name, aug) in enumerate(augmentations.items(), 1):
    augmented = aug(image=image)["image"]

    plt.subplot(4, 3, idx)
    plt.imshow(augmented)
    plt.title(name)
    plt.axis("off")

    if SAVE_OUTPUT:
        out_path = os.path.join(OUT_DIR, f"{name.replace(' ', '_')}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

plt.tight_layout()
plt.show()
