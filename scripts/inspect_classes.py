import os
import yaml
import xml.etree.ElementTree as ET

ROOT = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets"

DATASETS = {
    "roboflow_night": {
        "labels": os.path.join(ROOT, "NightPotholesDataset", "train", "labels"),
        "yaml": os.path.join(ROOT, "NightPotholesDataset", "data.yaml"),
        "type": "yolo"
    },
    "mwpd_multiweather": {
        "labels": os.path.join(ROOT, "MultiweatherPotholeDataset", "train", "labels"),
        "yaml": os.path.join(ROOT, "MultiweatherPotholeDataset", "data.yaml"),
        "type": "yolo"
    },
    "kaggle_multiview": {
        "labels": os.path.join(ROOT, "MultiviewPotholeDataset", "annotations"),
        "type": "voc"
    }
}


def inspect_yolo_labels(label_dir):
    ids = set()
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                if line.strip():
                    ids.add(int(line.split()[0]))
    return sorted(ids)


def load_yaml_names(yaml_path):
    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names", None)

    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return names
    else:
        return None


def inspect_voc_labels(xml_dir):
    classes = set()
    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(xml_dir, file))
        root = tree.getroot()
        for obj in root.findall("object"):
            classes.add(obj.find("name").text)
    return sorted(classes)


print("\n========== CLASS INSPECTION REPORT ==========\n")

for name, cfg in DATASETS.items():
    print(f"📂 Dataset: {name}")

    if cfg["type"] == "yolo":
        ids = inspect_yolo_labels(cfg["labels"])
        print(f"  YOLO class IDs found: {ids}")

        names = load_yaml_names(cfg.get("yaml", ""))
        if names:
            print("  Class mapping:")
            for k, v in names.items():
                print(f"    {k}: {v}")
        else:
            print("  ⚠ No data.yaml or names not found")

        if len(ids) > 1:
            print("  ⚠ WARNING: Multi-class dataset detected")

    elif cfg["type"] == "voc":
        classes = inspect_voc_labels(cfg["labels"])
        print(f"  VOC classes found: {classes}")

    print("-" * 50)
