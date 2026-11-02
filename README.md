# AI-Based Mall Occupancy Monitoring System

## 1. Project Overview

The AI-Based Mall Occupancy Monitoring System is a computer vision project that detects and counts people in mall images using the YOLOv8 object detection model.

The system analyzes mall images, estimates the number of people present, compares predictions with ground-truth counts, and displays occupancy statistics through an interactive Streamlit dashboard.

## 2. Objectives

- Analyze the mall crowd-counting dataset.
- Detect people using a pretrained YOLOv8 model.
- Estimate the number of people in each frame.
- Evaluate detection performance using MAE and RMSE.
- Analyze actual and detected occupancy trends.
- Display results through an interactive dashboard.

## 3. Technologies Used

- Python
- YOLOv8
- OpenCV
- NumPy
- Pandas
- SciPy
- Matplotlib
- Streamlit

## 4. Dataset Information

The project uses the Mall Dataset.

- Total images: 2,000
- Image dimensions: 640 × 480 pixels
- Image format: JPG
- Ground-truth occupancy information: MATLAB (.mat) files

The available dataset contains individual images rather than a video file. Therefore, video FPS is not available from the supplied files.

## 5. Methodology

### Step 1: Dataset Analysis
Analyzed image count, dimensions, image format and ground-truth information.

### Step 2: Person Detection
Used pretrained YOLOv8n to identify people in mall images. Only detections belonging to the person class were counted.

### Step 3: Model Evaluation
Compared YOLOv8 predictions with ground-truth counts using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Step 4: Occupancy Analytics
Calculated average, maximum and minimum occupancy and generated an occupancy trend graph.

### Step 5: Dashboard
Developed an interactive Streamlit dashboard to display occupancy statistics, detection results, model evaluation and frame-level analysis.

## 6. Results

| Metric | Result |
|---|---:|
| Total Frames | 2,000 |
| Average Actual Occupancy | 31.16 |
| Peak Actual Occupancy | 53 |
| Average YOLO Detection | 18.01 |
| Peak YOLO Detection | 34 |
| MAE | 13.15 |
| RMSE | 13.82 |

## 7. Limitations

The YOLOv8n model undercounts people in several frames. Small, overlapping and partially hidden people may be difficult to detect.

The current system analyzes the supplied dataset images and is not connected to a live CCTV camera.

## 8. Future Improvements

- Improve crowd detection accuracy.
- Evaluate other YOLO model versions.
- Add video-based occupancy monitoring.
- Integrate live CCTV feeds.
- Add alerts for high occupancy levels.

## 9. How to Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in the browser.

## 10. Conclusion

The project demonstrates how computer vision can be used for mall occupancy monitoring. YOLOv8 performs person detection, while Python-based analytics and Streamlit provide visual insights into crowd occupancy and model performance.