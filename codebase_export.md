# Codebase Export
**Source Directory:** `D:\Study Material\Sem VI\Projects\PBL2\RoadEye`

---

### File: `.gitignore`

```text
# =========================
# Python / Conda
# =========================
venv/
env/
.conda/
__pycache__/
*.py[cod]

# =========================
# VS Code
# =========================
.vscode/

# =========================
# Datasets & Outputs (DO NOT PUSH)
# =========================
Datasets/
runs/
assets/

# =========================
# Model weights
# =========================
*.pt
*.pth
*.onnx

# =========================
# Large numpy / data files
# =========================
*.npy
*.npz

# =========================
# Environment variables
# =========================
.env

# =========================
# OS files
# =========================
Thumbs.db
.DS_Store
```

### File: `export_codebase.py`

```python
import os
import json
import sqlite3
import argparse
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# Directories to completely ignore (Added ML logging & checkpoint folders)
EXCLUDE_DIRS = {
    'node_modules', 'venv', 'env', '.env', '.venv', '__pycache__',
    '.git', '.vscode', '.idea', 'dist', 'build', '.next',
    'coverage', '.pytest_cache', 'checkpoints', 'wandb', 'runs',
    'logs', 'tb_logs', 'huggingface', 'MedVoxelSampleDataset', 'data', 'models', 'outputs', 'results', 'figures', 'images', 'media', 'assets', ''
    'Datasets', 'notebooks', 'scripts', 'src', 'lib', 'external', 'third_party','assets', 'resources', 'static', 'public', 'cache', 'tmp', 'temp', 'backup', 'backups', 'old', 'archive', 'archives'
}

# File extensions to explicitly ignore grouped by category
EXCLUDE_EXTS = {
    # AI / ML Models, Weights, & Serialized Objects
    '.pt', '.pth', '.safetensors', '.ckpt', '.bin', '.onnx',
    '.tflite', '.pb', '.h5', '.hdf5', '.keras', '.pkl',
    '.joblib', '.gguf', '.mlmodel', '.t7', '.model', '.weights', '.npy', '.npz', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp', '.mp4', '.mp3', '.wav', '.flac', '.avi', '.mov',
    '.xml', '.txt', '.torchscript', '.torch', '.jsonl', '.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.tfrecord', '.nc', '.arrow', '.nii', '.nii.gz', '.dcm', '.dicom',

    # Heavy Data Formats
    '.parquet', '.tfrecord', '.nc', '.arrow', '.nii', '.nii.gz', '.dcm', '.dicom', '.cache', '.log', '.log.gz', '.hdf5', '.h5', '.hdf', '.hdf4', '.hdf5', '.h5ad', '.loom', '.zarr',
    '.jpgres', '.jpegres', '.pngres', '.bmpres', '.tifres', '.tiffres', '.webpres', '.gifres', '.svgres',

    # Images & Media
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
    '.mp4', '.mp3', '.wav', '.flac', '.avi', '.mov',

    # Archives & Documents
    '.pdf', '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz',

    # Compiled Binaries & Executables
    '.exe', '.dll', '.so', '.dylib', '.pyc', '.pyd', '.o', '.a', '.class',

    # Fonts
    '.woff', '.woff2', '.ttf', '.eot', '.otf'
}

# ==========================================
# SPECIAL FILE HANDLERS
# ==========================================

def extract_db_schema(db_path):
    """Connects to a SQLite database and extracts its table schemas."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schemas = cursor.fetchall()
        conn.close()

        schema_text = "\n\n".join([s[0] for s in schemas if s[0]])
        return schema_text if schema_text else "-- No tables found in database."
    except Exception as e:
        return f"-- Error reading SQLite DB schema: {e}"


def extract_ipynb_code(ipynb_path):
    """Reads a Jupyter notebook and extracts only the code from code cells."""
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        code_blocks = []
        for cell in data.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                if source.strip():
                    code_blocks.append(source)

        return "\n\n# --- Next Cell ---\n\n".join(code_blocks)
    except Exception as e:
        return f"# Error reading Jupyter Notebook: {e}"


def read_text_file(file_path):
    """Reads standard text/code files."""
    try:
        # Using errors='replace' ensures the script doesn't crash on weird encodings
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"// Error reading file text: {e}"


# ==========================================
# MAIN EXPORT LOGIC
# ==========================================

def get_markdown_language(ext):
    """Maps file extensions to markdown code block languages for syntax highlighting."""
    ext = ext.lower()
    map_dict = {
        '.py': 'python', '.js': 'javascript', '.jsx': 'jsx',
        '.ts': 'typescript', '.tsx': 'tsx', '.html': 'html',
        '.css': 'css', '.scss': 'scss', '.json': 'json',
        '.md': 'markdown', '.sql': 'sql', '.db': 'sql',
        '.sqlite': 'sql', '.ipynb': 'python', '.sh': 'bash',
        '.jinja': 'html', '.j2': 'html', '.yml': 'yaml', '.yaml': 'yaml',
        '.rs': 'rust', '.go': 'go', '.java': 'java', '.cpp': 'cpp', '.c': 'c'
    }
    return map_dict.get(ext, 'text')


def export_codebase(root_dir, output_file):
    root_path = Path(root_dir).resolve()
    output_path = Path(output_file).resolve()

    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write(f"# Codebase Export\n")
        out_f.write(f"**Source Directory:** `{root_path}`\n\n")
        out_f.write("---\n\n")

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Modify dirnames in-place to prevent os.walk from visiting excluded directories
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

            for file in filenames:
                file_path = Path(dirpath) / file

                # Skip the output file itself if it's inside the target directory
                if file_path.resolve() == output_path:
                    continue

                ext = file_path.suffix.lower()

                # Skip excluded extensions
                if ext in EXCLUDE_EXTS:
                    continue

                # Calculate relative path for clean display
                rel_path = file_path.relative_to(root_path)

                # Determine how to extract content based on extension
                if ext in ['.db', '.sqlite']:
                    content = extract_db_schema(file_path)
                elif ext == '.ipynb':
                    content = extract_ipynb_code(file_path)
                else:
                    content = read_text_file(file_path)

                # Skip completely empty files to save space
                if not content.strip():
                    continue

                # Write to the export file
                md_lang = get_markdown_language(ext)
                out_f.write(f"### File: `{rel_path}`\n\n")
                out_f.write(f"```{md_lang}\n")
                out_f.write(content)
                if not content.endswith('\n'):
                    out_f.write('\n')
                out_f.write("```\n\n")

    print(f"✅ Successfully exported codebase to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a codebase folder into a single Markdown file.")
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=".",
        help="The root directory to export (defaults to current directory)."
    )
    parser.add_argument(
        "-o", "--output",
        default="codebase_export.md",
        help="The name of the output Markdown file."
    )

    args = parser.parse_args()

    # Run the exporter
    export_codebase(args.target_dir, args.output)
```

### File: `gantest.ipynb`

```python
import pandas as pd
df = pd.read_parquet(r"C:\Users\parth\Downloads\test-00002-of-00008.parquet")
print(df.head())   
print(df.columns)
print(df.dtypes)
row = df.iloc[0]
print(type(row))
print(row)


# --- Next Cell ---

from PIL import Image
import io

row = df.iloc[0]

# Extract basecolor
img_bytes = row['basecolor']['bytes']

image = Image.open(io.BytesIO(img_bytes))
image.show()


# --- Next Cell ---

normal_bytes = row['normal']['bytes']
normal_img = Image.open(io.BytesIO(normal_bytes))
normal_img.show()


# --- Next Cell ---

import numpy as np

def bytes_to_image(byte_dict):
    return Image.open(io.BytesIO(byte_dict['bytes'])).convert("RGB")

base = bytes_to_image(row['basecolor'])
normal = bytes_to_image(row['normal'])
rough = bytes_to_image(row['roughness'])
metal = bytes_to_image(row['metallic'])

# Resize to same size
size = 256
base = base.resize((size, size))
normal = normal.resize((size, size))
rough = rough.resize((size, size))
metal = metal.resize((size, size))

# Create grid
atlas = Image.new("RGB", (size*2, size*2))
atlas.paste(base, (0, 0))
atlas.paste(normal, (size, 0))
atlas.paste(rough, (0, size))
atlas.paste(metal, (size, size))

atlas.show()
```

### File: `pbl_phase2.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RoadEye AI – End-to-End Workflow</title>

<style>
    body {
        margin: 0;
        padding: 0;
        font-family: "Inter", "Segoe UI", sans-serif;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }

    h1 {
        text-align: center;
        margin: 30px 0 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    h2 {
        text-align: center;
        font-weight: 400;
        color: #cfd9df;
        margin-bottom: 40px;
    }

    .workflow {
        max-width: 1200px;
        margin: auto;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        padding: 40px;
    }

    .step {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .step:hover {
        transform: translateY(-8px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    }

    .step::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 5px;
        width: 100%;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
    }

    .step-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #ffffff;
    }

    .step-desc {
        font-size: 14px;
        line-height: 1.6;
        color: #e0eafc;
    }

    footer {
        text-align: center;
        padding: 20px;
        font-size: 13px;
        color: #b0bec5;
    }

    .tag {
        display: inline-block;
        margin-top: 10px;
        padding: 4px 10px;
        font-size: 11px;
        border-radius: 20px;
        background: rgba(0,198,255,0.2);
        color: #9fe7ff;
    }
</style>
</head>

<body>

<h1>RoadEye AI – End-to-End System Workflow</h1>
<h2>Real-Time Road Damage Detection, Geo-Mapping & Civic Automation</h2>

<div class="workflow">

    <div class="step">
        <div class="step-title">1. Dataset Collection</div>
        <div class="step-desc">
            Gather pothole, crack, manhole datasets from multiple sources covering
            day/night, multi-weather, multi-view and motion scenarios.
        </div>
        <span class="tag">Data Acquisition</span>
    </div>

    <div class="step">
        <div class="step-title">2. Dataset Inspection & Format Analysis</div>
        <div class="step-desc">
            Inspect image formats (JPEG, PNG) and annotation formats
            (JSON, XML, TXT). Identify class labels and inconsistencies.
        </div>
        <span class="tag">Data Understanding</span>
    </div>

    <div class="step">
        <div class="step-title">3. Dataset Pre-processing</div>
        <div class="step-desc">
            Homogenize datasets into YOLO format. Perform resizing, normalization,
            class remapping and annotation cleanup.
        </div>
        <span class="tag">Preprocessing</span>
    </div>

    <div class="step">
        <div class="step-title">4. Data Augmentation</div>
        <div class="step-desc">
            Apply motion blur (Gaussian), low-light simulation, noise, weather effects
            and perspective changes to improve real-world robustness.
        </div>
        <span class="tag">Robust Training</span>
    </div>

    <div class="step">
        <div class="step-title">5. Dataset Fusion</div>
        <div class="step-desc">
            Combine regular, night-time, multi-weather and multi-angle datasets
            into a unified training set with balanced classes.
        </div>
        <span class="tag">Data Engineering</span>
    </div>

    <div class="step">
        <div class="step-title">6. Model Selection</div>
        <div class="step-desc">
            Choose suitable YOLO version (YOLOv8/YOLOv9/YOLOv10) based on
            accuracy, latency and mobile deployability.
        </div>
        <span class="tag">Model Design</span>
    </div>

    <div class="step">
        <div class="step-title">7. Model Training (PC)</div>
        <div class="step-desc">
            Train and fine-tune the YOLO model on a PC/GPU. Optimize hyperparameters
            for precision, recall and inference speed.
        </div>
        <span class="tag">Training</span>
    </div>

    <div class="step">
        <div class="step-title">8. Evaluation & Validation</div>
        <div class="step-desc">
            Validate model using test images/videos. Measure precision, recall,
            mAP, FPS and failure cases.
        </div>
        <span class="tag">Validation</span>
    </div>

    <div class="step">
        <div class="step-title">9. Model Conversion</div>
        <div class="step-desc">
            Convert trained model to ONNX / TensorFlow Lite for mobile deployment.
            Apply quantization and pruning if required.
        </div>
        <span class="tag">Optimization</span>
    </div>

    <div class="step">
        <div class="step-title">10. Deployment Strategy</div>
        <div class="step-desc">
            Choose between web-based inference (server-side) or native Android
            app (Kotlin) with on-device GPU/NNAPI acceleration.
        </div>
        <span class="tag">Deployment</span>
    </div>

    <div class="step">
        <div class="step-title">11. Camera & Mounting Setup</div>
        <div class="step-desc">
            Select stable phone mount and camera angle. Optimize shutter speed,
            frame skipping and vibration handling for bike-mounted usage.
        </div>
        <span class="tag">Hardware Integration</span>
    </div>

    <div class="step">
        <div class="step-title">12. Real-Time Inference Pipeline</div>
        <div class="step-desc">
            Capture live camera feed, perform inference, smooth detections,
            filter noise and log damage events in real time.
        </div>
        <span class="tag">Edge AI</span>
    </div>

    <div class="step">
        <div class="step-title">13. GIS Integration</div>
        <div class="step-desc">
            Attach GPS coordinates using Android Location Services. Assign label IDs,
            store in database and process spatial data using GeoPandas.
        </div>
        <span class="tag">Geo-Mapping</span>
    </div>

    <div class="step">
        <div class="step-title">14. Visualization & Analytics</div>
        <div class="step-desc">
            Display detections using Folium maps with markers, heatmaps,
            severity coloring and route-wise damage tracking.
        </div>
        <span class="tag">Visualization</span>
    </div>

    <div class="step">
        <div class="step-title">15. Deduplication & Quality Control</div>
        <div class="step-desc">
            Cluster detections spatially and temporally to avoid duplicate reports.
            Filter false positives and low-confidence detections.
        </div>
        <span class="tag">Reliability</span>
    </div>

    <div class="step">
        <div class="step-title">16. Autonomous Complaint Agent</div>
        <div class="step-desc">
            Use GenAI & Agentic AI to draft complaints using annotated snapshots,
            GPS data and severity. Provide UI to send via email, social media or portals.
        </div>
        <span class="tag">Automation</span>
    </div>

    <div class="step">
        <div class="step-title">17. User Dashboard</div>
        <div class="step-desc">
            Maintain drive history, detections, maps and complaint logs.
            Enable review, export and audit of collected data.
        </div>
        <span class="tag">Application Layer</span>
    </div>

    <div class="step">
        <div class="step-title">18. Monitoring & Future Scaling</div>
        <div class="step-desc">
            Monitor performance, retrain with new data, scale to multiple riders,
            cities and smart-infrastructure platforms.
        </div>
        <span class="tag">Scalability</span>
    </div>

</div>

<footer>
    © RoadEye AI | PBL-2 | B.Tech AIML | End-to-End Workflow Diagram
</footer>

</body>
</html>
```

