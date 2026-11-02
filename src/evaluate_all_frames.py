
from ultralytics import YOLO
from scipy.io import loadmat
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

# Dataset paths
dataset_path = Path("data/mall_dataset")
frames_path = dataset_path / "frames"

# Get all images in order
image_files = sorted(frames_path.glob("*.jpg"))

# Load ground-truth counts
ground_truth = loadmat(str(dataset_path / "mall_gt.mat"))
actual_counts = ground_truth["count"].flatten()

# Check dataset consistency
if len(image_files) != len(actual_counts):
    raise ValueError(
        f"Mismatch: {len(image_files)} images but "
        f"{len(actual_counts)} ground-truth counts."
    )

detected_counts = []
valid_actual_counts = []
valid_frame_names = []

print("\n========== FULL DATASET EVALUATION ==========")
print("Total frames:", len(image_files))
print("Starting detection...\n")

# Process every image
for i, image_path in enumerate(image_files):

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path.name)
        continue

    # Detect only people (class ID 0)
    results = model.predict(
        image,
        classes=[0],
        verbose=False
    )

    detected = len(results[0].boxes)
    actual = int(actual_counts[i])

    detected_counts.append(detected)
    valid_actual_counts.append(actual)
    valid_frame_names.append(image_path.name)

    # Show progress after every 100 frames
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(image_files)} frames")

# Convert counts into arrays
actual_values = np.array(valid_actual_counts)
detected_values = np.array(detected_counts)

# Calculate evaluation metrics
mae = np.mean(np.abs(actual_values - detected_values))
rmse = np.sqrt(np.mean((actual_values - detected_values) ** 2))

# Save frame-by-frame results
report = pd.DataFrame({
    "Frame": valid_frame_names,
    "Actual Count": actual_values,
    "Detected Count": detected_values
})

report_folder = Path("reports")
report_folder.mkdir(parents=True, exist_ok=True)

report.to_csv(
    report_folder / "full_evaluation.csv",
    index=False
)

# Display final results
print("\n========== FINAL EVALUATION RESULTS ==========")
print("Frames Evaluated:", len(detected_values))
print("MAE:", round(float(mae), 2))
print("RMSE:", round(float(rmse), 2))
print("Average Actual Count:", round(float(np.mean(actual_values)), 2))
print("Average Detected Count:", round(float(np.mean(detected_values)), 2))

print("\nResults saved to reports/full_evaluation.csv")
print("FULL DATASET EVALUATION COMPLETED!")
