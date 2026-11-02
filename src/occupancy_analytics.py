
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load evaluation results
data = pd.read_csv("reports/full_evaluation.csv")

actual = data["Actual Count"]
detected = data["Detected Count"]

print("========== OCCUPANCY ANALYTICS ==========")

print("\nACTUAL OCCUPANCY")
print("Average Occupancy:", round(actual.mean(), 2))
print("Peak Occupancy:", actual.max())
print("Minimum Occupancy:", actual.min())

print("\nYOLO DETECTED OCCUPANCY")
print("Average Detected:", round(detected.mean(), 2))
print("Peak Detected:", detected.max())
print("Minimum Detected:", detected.min())

# Create occupancy trend graph
plt.figure(figsize=(14, 6))

plt.plot(actual, label="Actual Occupancy", linewidth=1)
plt.plot(detected, label="YOLO Detected Occupancy", linewidth=1)

plt.title("Mall Occupancy Trend")
plt.xlabel("Frame Number")
plt.ylabel("Number of People")
plt.legend()
plt.grid(True)

# Save graph
output_folder = Path("reports")
output_folder.mkdir(exist_ok=True)

plt.savefig(output_folder / "occupancy_trend.png")
plt.close()

print("\nGraph saved to reports/occupancy_trend.png")
print("Occupancy Analytics Completed!")
