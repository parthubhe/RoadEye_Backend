from ultralytics import YOLO
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import os

# Define the path to your weights folder
weights_path = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights"
pt_file = os.path.join(weights_path, "best.pt")
ts_file = os.path.join(weights_path, "best.torchscript")

# 1. Load your model
print(f"Loading model from: {pt_file}")
model = YOLO(pt_file)

# 2. Export to TorchScript
# This creates 'best.torchscript' inside the 'weights' folder defined above
model.export(format="torchscript", imgsz=320, optimize=True)

# 3. Load the exported script to apply Mobile Optimizations
# FIX: Use the full path to where the export was actually saved
print(f"Loading for optimization from: {ts_file}")
ts_model = torch.jit.load(ts_file)

# 4. Apply Mobile Optimization (Lite Interpreter)
optimized_model = optimize_for_mobile(ts_model)

# 5. Save the fast version
# This saves 'best_optimized.torchscript' in your CURRENT folder (where you run the script from)
output_name = "best_optimized.torchscript"
optimized_model._save_for_lite_interpreter(output_name)

print(f"✅ Done! Created {output_name}")
print("Move this file to your Android Studio assets folder.")