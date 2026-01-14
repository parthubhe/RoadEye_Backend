import os
import xml.etree.ElementTree as ET

ROOT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiviewPotholeDataset"
VOC_DIR = os.path.join(ROOT, "annotations")
OUT_DIR = os.path.join(ROOT, "labels_yolo")

os.makedirs(OUT_DIR, exist_ok=True)

CLASS_MAP = {
    "pothole": 1
}

for file in os.listdir(VOC_DIR):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(VOC_DIR, file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in CLASS_MAP:
            continue

        cid = CLASS_MAP[cls]
        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    out_path = os.path.join(OUT_DIR, file.replace(".xml", ".txt"))
    with open(out_path, "w") as f:
        f.write("\n".join(yolo_lines))

print("✅ Kaggle VOC converted to YOLO (pothole=1)")
