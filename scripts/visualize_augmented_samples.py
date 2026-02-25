import os
import cv2
import random
import matplotlib.pyplot as plt

DATASET = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\RoadEyeUnified_SingleClass_Aug"
SPLIT = "train"
NUM_SAMPLES = 20

img_dir = os.path.join(DATASET, "images", SPLIT)
lbl_dir = os.path.join(DATASET, "labels", SPLIT)

def draw_boxes(img, boxes):
    h, w, _ = img.shape
    for box in boxes:
        x, y, bw, bh = map(float, box)
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    return img

images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

sample_imgs = random.sample(images, NUM_SAMPLES)

for img_name in sample_imgs:

    base = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(lbl_dir, base + ".txt")

    if not os.path.exists(lbl_path):
        continue

    img = cv2.imread(os.path.join(img_dir, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            boxes.append(parts[1:])

    img = draw_boxes(img, boxes)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.title(img_name)
    plt.axis("off")
    plt.show()
