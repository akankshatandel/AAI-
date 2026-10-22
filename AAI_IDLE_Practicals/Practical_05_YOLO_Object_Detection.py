# PRACTICAL 5: Object detection using YOLOv8
# Install: python -m pip install ultralytics opencv-python matplotlib
# Internet is required on the first run to download the model and sample image.

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import urllib.request
import numpy as np

image_url = "https://ultralytics.com/images/bus.jpg"
image_path = "sample_bus.jpg"

print("Downloading sample image...")
urllib.request.urlretrieve(image_url, image_path)

model = YOLO("yolov8n.pt")
image = cv2.imread(image_path)
if image is None:
    raise RuntimeError("Could not load sample image. Check internet access.")

results = model(image, verbose=False)
annotated = results[0].plot()

plt.figure(figsize=(9, 7))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("YOLOv8 Object Detection")
plt.show()

print("Detected objects:")
names = model.names
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    print(names[class_id], "- confidence:", round(confidence, 2))
