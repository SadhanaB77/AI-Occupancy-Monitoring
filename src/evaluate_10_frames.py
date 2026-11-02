
from ultralytics import YOLO
from scipy.io import loadmat
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Load YOLO model
model = YOLO("yolov8n.pt")

# Dataset paths
dataset_path = Path("data/mall_dataset")
frames_path = dataset_path / "frames"

# Get first 10 images
image_files = sorted(frames_path.glob("*.jpg"))[:10]

# Load actual people counts
ground_truth = loadmat(str(dataset_path / "mall_gt.mat"))
actual_counts = ground_truth["count"].flatten()

detected_counts = []

print("========== 10 FRAME EVALUATION ==========")

for i, image_path in enumerate(image_files):

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path.name)
        continue

    # Detect people only
    results = model.predict(
        image,
        classes=[0],
        verbose=False
    )

    # Count detected people
    detected = len(results[0].boxes)
    actual = int(actual_counts[i])

    detected_counts.append(detected)

    print(
        f"Frame {i + 1}: "
        f"Actual = {actual}, "
        f"Detected = {detected}"
    )

# Calculate evaluation metrics
actual_values = actual_counts[:len(detected_counts)]
detected_values = np.array(detected_counts)

mae = np.mean(np.abs(actual_values - detected_values))
rmse = np.sqrt(np.mean((actual_values - detected_values) ** 2))

print("\n========== EVALUATION RESULTS ==========")
print("Frames Evaluated:", len(detected_counts))
print("MAE:", round(float(mae), 2))
print("RMSE:", round(float(rmse), 2))

# Save results as CSV
report = pd.DataFrame({
    "Frame": [image_files[i].name for i in range(len(detected_counts))],
    "Actual Count": actual_values,
    "Detected Count": detected_values
})

report_path = Path("reports/first_10_evaluation.csv")
report_path.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(report_path, index=False)

print("\nResults saved to:", report_path)
