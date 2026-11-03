
# ============================================================
# MALLSIGHT AI - AI-BASED OCCUPANCY MONITORING SYSTEM
# ============================================================

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from ultralytics import YOLO

import matplotlib.pyplot as plt

from src.people_tracking import (
    track_line_crossing,
    count_people_in_region
)


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="MallSight AI",
    page_icon="🏬",
    layout="wide"
)


# ---------------- PROJECT PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent

REPORTS_DIR = BASE_DIR / "reports"
DETECTION_DIR = REPORTS_DIR / "detection"
SAMPLE_FRAMES_DIR = REPORTS_DIR / "sample_frames"

MODEL_PATH = BASE_DIR / "yolov8n.pt"

REPORTS_DIR.mkdir(exist_ok=True)
DETECTION_DIR.mkdir(exist_ok=True)


# ---------------- LOAD YOLO MODEL ----------------

@st.cache_resource
def load_model():

    return YOLO(str(MODEL_PATH))


# ---------------- SIDEBAR ----------------

st.sidebar.title("🏬 MallSight AI")

st.sidebar.write(
    "AI-Based Occupancy Monitoring System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Detection Results",
        "Frame Analysis",
        "Upload & Detect",
        "Line Crossing",
        "Area Counting"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### About the Project")

st.sidebar.write(
    """
    MallSight AI uses computer vision to monitor
    people in shopping mall environments.

    **Technologies:**
    - Python
    - OpenCV
    - YOLOv8
    - Streamlit
    """
)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.title("🏬 MallSight AI")

    st.subheader(
        "AI-Based Occupancy Monitoring System"
    )

    st.write(
        """
        MallSight AI is a computer vision-based system
        designed to detect and monitor people in shopping
        mall environments.

        The system uses YOLOv8 for person detection and
        provides occupancy analysis through an interactive
        dashboard.
        """
    )

    st.divider()

    st.markdown("### Project Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("#### 👥 Person Detection")

        st.write(
            "Detect people in mall images and video frames "
            "using YOLOv8."
        )

    with col2:

        st.markdown("#### 📊 Occupancy Analytics")

        st.write(
            "Analyse people counts and understand occupancy "
            "trends."
        )

    with col3:

        st.markdown("#### 🚶 Movement Monitoring")

        st.write(
            "Monitor people crossing a virtual line and "
            "count people inside a selected area."
        )

    st.divider()

    st.markdown("### Dataset Information")

    dataset_path = BASE_DIR / "data" / "mall_dataset"

    if dataset_path.exists():

        image_files = list(dataset_path.glob("*.jpg"))

        st.metric(
            "Dataset Images",
            len(image_files)
        )

    else:

        st.info(
            "Dataset folder is not available in this deployment. "
            "You can still use the dashboard with uploaded videos."
        )

    st.markdown("### Model Information")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Detection Model",
            "YOLOv8n"
        )

    with col2:

        st.metric(
            "Detection Class",
            "Person"
        )

    st.divider()

    st.markdown("### Navigation Guide")

    st.write(
        """
        - **Overview:** Project introduction and features.
        - **Detection Results:** View existing detection results.
        - **Frame Analysis:** Explore frame-wise people counts.
        - **Upload & Detect:** Upload an image or video for detection.
        - **Line Crossing:** Count people moving across a virtual line.
        - **Area Counting:** Count people inside a selected screen region.
        """
    )

# ============================================================
# PAGE 2: DETECTION RESULTS
# ============================================================

elif page == "Detection Results":

    st.title("📊 Detection Results")

    st.write(
        "View the person detection results generated from "
        "the Mall Dataset."
    )

    evaluation_file = REPORTS_DIR / "full_evaluation.csv"

    if evaluation_file.exists():

        df = pd.read_csv(evaluation_file)

        if not df.empty:

            required_columns = [
                "Actual Count",
                "Detected Count"
            ]

            if all(col in df.columns for col in required_columns):

                mae = np.mean(
                    np.abs(
                        df["Actual Count"] - df["Detected Count"]
                    )
                )

                rmse = np.sqrt(
                    np.mean(
                        (
                            df["Actual Count"] -
                            df["Detected Count"]
                        ) ** 2
                    )
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Frames",
                    len(df)
                )

                col2.metric(
                    "MAE",
                    f"{mae:.2f}"
                )

                col3.metric(
                    "RMSE",
                    f"{rmse:.2f}"
                )

                st.divider()

                st.markdown("### Actual vs Detected People")

                chart_df = df[
                    ["Actual Count", "Detected Count"]
                ]

                st.line_chart(chart_df)

                st.markdown("### Evaluation Data")

                st.dataframe(
                    df,
                    width="stretch",
                    hide_index=True
                )

                csv_data = df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Evaluation CSV",
                    data=csv_data,
                    file_name="full_evaluation.csv",
                    mime="text/csv"
                )

            else:

                st.warning(
                    "The evaluation CSV does not contain the "
                    "required Actual Count and Detected Count columns."
                )

        else:

            st.info("The evaluation CSV is empty.")

    else:

        st.info(
            "Evaluation results are not available. "
            "Please add reports/full_evaluation.csv."
        )

    st.divider()

    st.markdown("### Detection Sample Image")

    sample_image = DETECTION_DIR / "detected_people.jpg"

    if sample_image.exists():

        st.image(
            str(sample_image),
            caption="YOLOv8 Person Detection Result",
            width="stretch"
        )

    else:

        st.info("Detection sample image is not available.")


# ============================================================
# PAGE 3: FRAME ANALYSIS
# ============================================================

elif page == "Frame Analysis":

    st.title("🖼️ Frame Analysis")

    st.write(
        "Explore the people count and detection results "
        "for individual frames."
    )

    evaluation_file = REPORTS_DIR / "full_evaluation.csv"

    if evaluation_file.exists():

        df = pd.read_csv(evaluation_file)

        if not df.empty:

            st.markdown("### Select Frame")

            frame_number = st.slider(
                "Choose a frame number",
                min_value=1,
                max_value=len(df),
                value=1
            )

            selected_row = df.iloc[frame_number - 1]

            col1, col2 = st.columns(2)

            col1.metric(
                "Actual People",
                int(selected_row["Actual Count"])
            )

            col2.metric(
                "Detected People",
                int(selected_row["Detected Count"])
            )

            st.divider()

            st.markdown("### Frame Detection Image")

            frame_image = (
                DETECTION_DIR / f"frame_{frame_number}.jpg"
            )

            if frame_image.exists():

                st.image(
                    str(frame_image),
                    caption=f"Frame {frame_number}",
                    width="stretch"
                )

            else:

                st.info(
                    "The individual detection image for this "
                    "frame is not available."
                )

            st.divider()

            st.markdown("### Frame-wise Comparison")

            comparison_df = df.copy()

            if "Frame" not in comparison_df.columns:

                comparison_df.insert(
                    0,
                    "Frame",
                    range(1, len(comparison_df) + 1)
                )

            st.line_chart(
                comparison_df.set_index("Frame")[
                    ["Actual Count", "Detected Count"]
                ]
            )

            st.markdown("### All Frame Records")

            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True
            )

        else:

            st.info("No frame records found.")

    else:

        st.info(
            "Frame evaluation data is not available. "
            "Please add reports/full_evaluation.csv."
        )

    st.divider()

    st.markdown("### Sample Frames")

    if SAMPLE_FRAMES_DIR.exists():

        sample_images = sorted(
            SAMPLE_FRAMES_DIR.glob("*.jpg")
        )

        if sample_images:

            columns = st.columns(3)

            for index, image_path in enumerate(sample_images):

                with columns[index % 3]:

                    st.image(
                        str(image_path),
                        caption=image_path.name,
                        width="stretch"
                    )

        else:

            st.info("No sample frame images found.")

    else:

        st.info("Sample frames folder is not available.")

# ============================================================
# PAGE 4: UPLOAD & DETECT
# ============================================================

elif page == "Upload & Detect":

    st.title("📤 Upload & Detect")

    st.write(
        "Upload an image or video to detect people using YOLOv8."
    )

    upload_type = st.radio(
        "Choose upload type",
        ["Image", "Video"],
        horizontal=True
    )

    # ---------------- IMAGE DETECTION ----------------

    if upload_type == "Image":

        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )

        if uploaded_image is not None:

            image = Image.open(uploaded_image).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                width="stretch"
            )

            if st.button("Detect People in Image"):

                try:

                    with st.spinner("Detecting people..."):

                        model = load_model()

                        result = model.predict(
                            source=image,
                            classes=[0],
                            conf=0.25,
                            verbose=False
                        )[0]

                        person_count = len(result.boxes)

                        annotated_image = result.plot()

                        annotated_image = cv2.cvtColor(
                            annotated_image,
                            cv2.COLOR_BGR2RGB
                        )

                    st.success("Image detection completed!")

                    st.metric(
                        "People Detected",
                        person_count
                    )

                    st.image(
                        annotated_image,
                        caption="Person Detection Result",
                        width="stretch"
                    )

                    image_bytes = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(
                            annotated_image,
                            cv2.COLOR_RGB2BGR
                        )
                    )[1].tobytes()

                    st.download_button(
                        "Download Detection Image",
                        data=image_bytes,
                        file_name="mall_detection_result.jpg",
                        mime="image/jpeg"
                    )

                except Exception as e:

                    st.error(f"Image detection failed: {e}")

    # ---------------- VIDEO DETECTION ----------------

    elif upload_type == "Video":

        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload"
        )

        if uploaded_video is not None:

            st.video(uploaded_video)

            frame_interval = st.slider(
                "Process every Nth frame",
                min_value=1,
                max_value=10,
                value=1,
                help="1 means every frame is processed."
            )

            if st.button("Detect People in Video"):

                input_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix or ".mp4"
                )

                input_temp.write(uploaded_video.getbuffer())
                input_temp.close()

                output_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_path = output_temp.name
                output_temp.close()

                cap = None
                writer = None

                try:

                    model = load_model()

                    cap = cv2.VideoCapture(input_temp.name)

                    if not cap.isOpened():

                        st.error("Unable to read this video.")

                    else:

                        fps = cap.get(cv2.CAP_PROP_FPS)

                        if fps <= 0:
                            fps = 25.0

                        width = int(
                            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        )

                        height = int(
                            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        )

                        total_frames = int(
                            cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        )

                        output_fps = fps / frame_interval

                        writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            (width, height)
                        )

                        if not writer.isOpened():

                            st.error(
                                "Unable to create the processed video."
                            )

                        else:

                            progress = st.progress(0)
                            status = st.empty()

                            frame_number = 0
                            counts = []

                            with st.spinner(
                                "Processing video frames..."
                            ):

                                while True:

                                    success, frame = cap.read()

                                    if not success:
                                        break

                                    if frame_number % frame_interval == 0:

                                        result = model.predict(
                                            source=frame,
                                            classes=[0],
                                            conf=0.25,
                                            verbose=False
                                        )[0]

                                        person_count = len(result.boxes)

                                        counts.append(person_count)

                                        annotated_frame = result.plot()

                                        writer.write(annotated_frame)

                                    frame_number += 1

                                    if total_frames > 0:

                                        progress.progress(
                                            min(
                                                frame_number / total_frames,
                                                1.0
                                            )
                                        )

                                    if frame_number % 30 == 0:

                                        status.write(
                                            f"Processed {frame_number} "
                                            f"of {total_frames} frames"
                                        )

                            cap.release()
                            cap = None

                            writer.release()
                            writer = None

                            progress.progress(1.0)

                            status.success(
                                "Video processing completed!"
                            )

                            if counts:

                                results_df = pd.DataFrame({
                                    "Processed Frame": range(
                                        1, len(counts) + 1
                                    ),
                                    "Detected People": counts
                                })

                                st.success(
                                    "Video detection completed!"
                                )

                                col1, col2, col3 = st.columns(3)

                                col1.metric(
                                    "Processed Frames",
                                    len(counts)
                                )

                                col2.metric(
                                    "Average People",
                                    f"{np.mean(counts):.2f}"
                                )

                                col3.metric(
                                    "Peak People",
                                    int(max(counts))
                                )

                                st.markdown("### Processed Video")

                                with open(output_path, "rb") as video_file:

                                    video_bytes = video_file.read()

                                st.video(video_bytes)

                                st.download_button(
                                    "Download Processed Video",
                                    data=video_bytes,
                                    file_name="mall_processed_video.mp4",
                                    mime="video/mp4"
                                )

                                st.markdown(
                                    "### Frame-wise People Count"
                                )

                                st.dataframe(
                                    results_df,
                                    width="stretch",
                                    hide_index=True
                                )

                                st.markdown(
                                    "### Video Occupancy Trend"
                                )

                                st.line_chart(
                                    results_df.set_index(
                                        "Processed Frame"
                                    )
                                )

                                csv_data = results_df.to_csv(
                                    index=False
                                ).encode("utf-8")

                                st.download_button(
                                    "Download Frame Count CSV",
                                    data=csv_data,
                                    file_name="video_occupancy_results.csv",
                                    mime="text/csv"
                                )

                            else:

                                st.warning(
                                    "No frames could be processed."
                                )

                except Exception as e:

                    st.error(
                        f"Video detection failed: {e}"
                    )

                finally:

                    if cap is not None:
                        cap.release()

                    if writer is not None:
                        writer.release()

                    if os.path.exists(input_temp.name):
                        os.remove(input_temp.name)

                    # Output file is kept for Streamlit display.

# ============================================================
# PAGE 5: LINE CROSSING
# ============================================================

elif page == "Line Crossing":

    st.title("🚶 Line Crossing Detection")

    st.write(
        "Count people moving from top to bottom and "
        "bottom to top by crossing a virtual horizontal line."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="line_crossing_video"
    )

    line_position = st.slider(
        "Virtual line position (%)",
        min_value=10,
        max_value=90,
        value=50,
        step=5
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Start Line Crossing Detection"):

            input_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_video.name).suffix or ".mp4"
            )

            input_temp.write(uploaded_video.getbuffer())
            input_temp.close()

            try:

                with st.spinner(
                    "Tracking people and detecting line crossings..."
                ):

                    result = track_line_crossing(
                        video_path=input_temp.name,
                        output_path=str(
                            REPORTS_DIR / "line_crossing_output.mp4"
                        ),
                        line_position=line_position / 100
                    )

                st.success("Line crossing detection completed!")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Top to Bottom",
                    result["top_to_bottom"]
                )

                col2.metric(
                    "Bottom to Top",
                    result["bottom_to_top"]
                )

                st.markdown("### Processed Video")

                output_path = result["output_path"]

                if os.path.exists(output_path):

                    with open(output_path, "rb") as video_file:
                        video_bytes = video_file.read()

                    st.video(video_bytes)

                    st.download_button(
                        "Download Line Crossing Video",
                        data=video_bytes,
                        file_name="line_crossing_output.mp4",
                        mime="video/mp4"
                    )

                st.markdown("### Crossing Summary")

                summary_df = pd.DataFrame(result["summary"])

                st.dataframe(
                    summary_df,
                    width="stretch",
                    hide_index=True
                )

                csv_data = summary_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Crossing Summary CSV",
                    data=csv_data,
                    file_name="line_crossing_summary.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Line crossing detection failed: {e}"
                )

            finally:

                if os.path.exists(input_temp.name):
                    os.remove(input_temp.name)


# ============================================================
# PAGE 6: AREA COUNTING
# ============================================================

elif page == "Area Counting":

    st.title("📍 Specific Area People Counting")

    st.write(
        "Select a rectangular region in the video. "
        "The system counts people whose bounding-box centre "
        "is inside the selected region."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="area_counting_video"
    )

    st.markdown("### Select Region of Interest (ROI)")

    col1, col2 = st.columns(2)

    with col1:

        x1 = st.slider(
            "Left boundary (%)",
            min_value=0,
            max_value=90,
            value=25
        )

        y1 = st.slider(
            "Top boundary (%)",
            min_value=0,
            max_value=90,
            value=25
        )

    with col2:

        x2 = st.slider(
            "Right boundary (%)",
            min_value=10,
            max_value=100,
            value=75
        )

        y2 = st.slider(
            "Bottom boundary (%)",
            min_value=10,
            max_value=100,
            value=75
        )

    if x1 >= x2 or y1 >= y2:

        st.warning(
            "Invalid region. Right boundary must be greater than "
            "left boundary, and bottom boundary must be greater than top."
        )

    elif uploaded_video is not None:

        st.caption(
            f"Selected region: X ({x1}% - {x2}%), "
            f"Y ({y1}% - {y2}%)"
        )

        st.video(uploaded_video)

        if st.button("Start Area Counting"):

            input_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_video.name).suffix or ".mp4"
            )

            input_temp.write(uploaded_video.getbuffer())
            input_temp.close()

            try:

                with st.spinner(
                    "Detecting people inside the selected area..."
                ):

                    result = count_people_in_region(
                        video_path=input_temp.name,
                        roi=(
                            x1 / 100,
                            y1 / 100,
                            x2 / 100,
                            y2 / 100
                        ),
                        output_path=str(
                            REPORTS_DIR / "area_counting_output.mp4"
                        )
                    )

                st.success("Area counting completed!")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "People in Last Frame",
                    result["last_frame_count"]
                )

                col2.metric(
                    "Average People",
                    f"{result['average']:.2f}"
                )

                col3.metric(
                    "Peak People",
                    result["peak"]
                )

                st.markdown("### Processed Video")

                output_path = result["output_path"]

                if os.path.exists(output_path):

                    with open(output_path, "rb") as video_file:
                        video_bytes = video_file.read()

                    st.video(video_bytes)

                    st.download_button(
                        "Download Area Counting Video",
                        data=video_bytes,
                        file_name="area_counting_output.mp4",
                        mime="video/mp4"
                    )

                # -------- CORRECTED AREA COUNT TABLE --------

                frame_counts = result["frame_counts"]

                trend_df = pd.DataFrame({
                    "Frame": range(1, len(frame_counts) + 1),
                    "People Inside Area": frame_counts
                })

                st.markdown(
                    "### People Count Inside Selected Area"
                )

                st.dataframe(
                    trend_df,
                    width="stretch",
                    hide_index=True
                )

                # -------- CORRECTED AREA OCCUPANCY TREND --------

                st.markdown("### Area Occupancy Trend")

                if not trend_df.empty:

                    st.line_chart(
                        trend_df.set_index("Frame")
                    )

                else:

                    st.info(
                        "No frame count data is available for the chart."
                    )

                csv_data = trend_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Area Counting CSV",
                    data=csv_data,
                    file_name="area_counting_results.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Area counting failed: {e}"
                )

            finally:

                if os.path.exists(input_temp.name):
                    os.remove(input_temp.name)

    else:

        st.info(
            "Upload a video to start area counting."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MallSight AI | AI-Based Occupancy Monitoring System"
)

st.caption(
    "Developed using Python, OpenCV, YOLOv8 and Streamlit"
)
