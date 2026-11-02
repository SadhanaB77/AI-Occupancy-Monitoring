
import cv2
import os
from pathlib import Path
from scipy.io import loadmat

# Dataset folder
dataset_path = Path("data/mall_dataset")

# Frames folder
frames_path = dataset_path / "frames"

# Get all JPG images
image_files = sorted(frames_path.glob("*.jpg"))

print("========== DATASET ANALYSIS ==========")

# Total images
total_images = len(image_files)
print("Total Images:", total_images)

# Check whether images exist
if total_images > 0:

    # Read the first image
    image = cv2.imread(str(image_files[0]))

    if image is not None:
        height, width, channels = image.shape

        print("\n========== IMAGE INFORMATION ==========")
        print("First Image:", image_files[0].name)
        print("Image Width:", width, "pixels")
        print("Image Height:", height, "pixels")
        print("Color Channels:", channels)
        print("Image Format: JPG")

        # Save sample images
        sample_folder = Path("reports/sample_frames")
        sample_folder.mkdir(parents=True, exist_ok=True)

        for i in range(min(5, total_images)):
            sample = cv2.imread(str(image_files[i]))

            if sample is not None:
                output_path = sample_folder / image_files[i].name
                cv2.imwrite(str(output_path), sample)

        print("\n5 sample images saved in reports/sample_frames")

    else:
        print("Could not read the first image.")

else:
    print("No JPG images found!")

# Inspect MATLAB files
print("\n========== MATLAB FILE INFORMATION ==========")

mat_files = ["mall_feat.mat", "mall_gt.mat", "perspective_roi.mat"]

for filename in mat_files:

    file_path = dataset_path / filename

    if file_path.exists():
        data = loadmat(str(file_path))

        print("\nFile:", filename)

        for key, value in data.items():
            if not key.startswith("__"):
                print("Variable:", key)
                print("Shape:", value.shape)

    else:
        print(filename, "not found")

print("\n========== ANALYSIS COMPLETED ==========")


# Step 8: Ground Truth Analysis

import numpy as np

print("\n========== GROUND TRUTH ANALYSIS ==========")

# Load ground truth file
ground_truth = loadmat(str(dataset_path / "mall_gt.mat"))

# Extract actual people counts
actual_counts = ground_truth["count"].flatten()

# Calculate statistics
minimum_count = np.min(actual_counts)
maximum_count = np.max(actual_counts)
average_count = np.mean(actual_counts)

print("Total Count Entries:", len(actual_counts))
print("Minimum People Count:", minimum_count)
print("Maximum People Count:", maximum_count)
print("Average People Count:", round(average_count, 2))

# Display first 10 frame counts
print("\nFirst 10 Frame Counts:")

for i in range(10):
    print(f"Frame {i + 1}: {actual_counts[i]} people")

print("\nGround Truth Analysis Completed!")
