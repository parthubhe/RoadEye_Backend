from ultralytics import YOLO
import os

# 1. Load your trained model
# Replace with your actual path if different
model_path = r"D:\Study Material\Sem VI\Projects\PBL2\RoadEye\runs\detect\runs\RoadEye\yolo11_multiclass_rtx4070\weights\best.pt"
model = YOLO(model_path)

# 2. Export specifically to 320x320 TFLite
# The 'nms=True' argument is optional but helpful if your version supports it. 
# We stick to standard export to be safe.
print("⚡ Exporting 320x320 TFLite model...")
model.export(format="tflite", imgsz=320, half=True)

# 3. Rename and Move instructions
print("\n✅ EXPORT DONE.")
print("👉 Go to the weights folder.")
print("👉 Find 'best_saved_model/best_float16.tflite' (or similar).")
print("👉 RENAME it to 'yolo320.tflite'.")
print("👉 COPY 'yolo320.tflite' to your Android 'app/src/main/assets/' folder.")