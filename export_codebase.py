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