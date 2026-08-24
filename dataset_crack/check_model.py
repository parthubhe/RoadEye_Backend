import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import CBAM
tasks.CBAM = CBAM

from ultralytics import YOLO

model = YOLO("yolo26n-cbam.yaml").load("yolo26n.pt")
model.info()
print(type(model.model.model[17]).__name__)