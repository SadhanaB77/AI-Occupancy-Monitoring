
from ultralytics import YOLO
import cv2
from pathlib import Path

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Select one mall image
image_path = "data/mall_dataset/frames/seq_000001.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# Detect objects
results = model(image)

# Count only people (class ID 0)
person_count = 0

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id == 0:
            person_count += 1

# Draw detection boxes
annotated_image = results[0].plot()

# Save output image
output_folder = Path("reports/detection")
output_folder.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(output_folder / "detected_people.jpg"),
    annotated_image
)

print("Person Detection Completed!")
print("Detected People:", person_count)
print("Output saved in reports/detection/detected_people.jpg")
