import os
from pathlib import Path
from datetime import datetime


def export_python_codebase(source_folder, output_file):
    source_path = Path(source_folder)

    print(f"Scanning folder: {source_path.resolve()}")

    if not source_path.exists() or not source_path.is_dir():
        print("❌ Invalid source folder.")
        return

    py_files = sorted(source_path.rglob("*.py"))

    print(f"Found {len(py_files)} Python files.")

    if not py_files:
        print("❌ No .py files found.")
        return

    output_path = Path(output_file).resolve()

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write("=" * 80 + "\n")
        outfile.write("PYTHON CODEBASE EXPORT\n")
        outfile.write(f"Source Folder: {source_path.resolve()}\n")
        outfile.write(f"Exported On : {datetime.now()}\n")
        outfile.write("=" * 80 + "\n\n")

        for file_path in py_files:
            relative_path = file_path.relative_to(source_path)

            separator = "=" * 80
            outfile.write(f"{separator}\n")
            outfile.write(f"FILE: {relative_path}\n")
            outfile.write(f"{separator}\n\n")

            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content.strip() + "\n\n")
            except Exception as e:
                outfile.write(f"[Error reading file: {e}]\n\n")

    print(f"✅ Export completed successfully!")
    print(f"📄 File saved at: {output_path}")


if __name__ == "__main__":
    source_folder = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\scripts"
    output_file = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\codebase_export.txt"

    export_python_codebase(source_folder, output_file)
