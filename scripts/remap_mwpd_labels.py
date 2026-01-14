import os

LABEL_DIR = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\Datasets\MultiweatherPotholeDataset\valid\labels"

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(LABEL_DIR, file)
    new_lines = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # remap class 0 → 1
            parts[0] = "1"
            new_lines.append(" ".join(parts))

    with open(path, "w") as f:
        f.write("\n".join(new_lines))

print("✅ MWPD labels remapped to pothole=1")
