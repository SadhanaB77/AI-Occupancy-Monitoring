
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tempfile
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="MallSight AI | Occupancy Dashboard",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #F5FAF9;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #E6F4F1;
    border-right: 1px solid #D4E8E3;
}
h1, h2, h3 {
    color: #164E45 !important;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCEBE7;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(20, 90, 75, 0.05);
}
[data-testid="stMetricLabel"] {
    color: #648078;
}
[data-testid="stMetricValue"] {
    color: #087F6B;
    font-weight: 700;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 14px;
}
.stButton > button {
    background-color: #087F6B;
    color: white;
    border-radius: 8px;
    border: none;
}
.stButton > button:hover {
    background-color: #066354;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD EXISTING DATA ----------------

report_path = Path("reports/full_evaluation.csv")
sample_image = Path("reports/detection/detected_people.jpg")

if not report_path.exists():
    st.error("Evaluation report not found. Run the full evaluation first.")
    st.stop()

data = pd.read_csv(report_path)

required_columns = ["Frame", "Actual Count", "Detected Count"]

if not all(column in data.columns for column in required_columns):
    st.error("The evaluation CSV does not contain the expected columns.")
    st.stop()

actual = data["Actual Count"]
detected = data["Detected Count"]

mae = (actual - detected).abs().mean()
rmse = (((actual - detected) ** 2).mean()) ** 0.5


# ---------------- LOAD YOLO MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("## 🏬 MallSight AI")
    st.caption("OCCUPANCY MONITORING SYSTEM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Detection Results",
            "Frame Analysis",
            "Upload & Detect"
        ]
    )

    st.divider()

    st.markdown("### Dataset Information")
    st.write("📁 Mall Dataset")
    st.write(f"🖼️ Total Frames: {len(data):,}")
    st.write("📐 Image Size: 640 × 480")
    st.write("🤖 Model: YOLOv8n")

    st.divider()

    st.caption("AI-Based Mall Occupancy Monitoring")


# ---------------- HEADER ----------------

st.markdown(
    "<h1>🏬 Mall Occupancy Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:#648078; font-size:16px;'>"
    "Intelligent crowd monitoring and occupancy analytics"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.subheader("📊 Occupancy Overview")

    st.markdown("#### Actual Mall Occupancy")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Occupancy", f"{actual.mean():.2f}")
    c2.metric("Peak Occupancy", int(actual.max()))
    c3.metric("Minimum Occupancy", int(actual.min()))

    st.markdown("#### YOLO Detected Occupancy")

    c4, c5, c6 = st.columns(3)
    c4.metric("Average Detected", f"{detected.mean():.2f}")
    c5.metric("Peak Detected", int(detected.max()))
    c6.metric("Minimum Detected", int(detected.min()))

    st.markdown("#### Model Performance")

    c7, c8 = st.columns(2)
    c7.metric("Mean Absolute Error", f"{mae:.2f}")
    c8.metric("Root Mean Squared Error", f"{rmse:.2f}")

    st.divider()

    st.subheader("📈 Mall Occupancy Trend")

    with st.container(border=True):

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            actual.values,
            label="Actual Occupancy",
            color="#087F6B",
            linewidth=1.8
        )

        ax.plot(
            detected.values,
            label="YOLO Detected",
            color="#F59E0B",
            linewidth=1.5
        )

        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Number of People")
        ax.set_title("Actual vs YOLO Detected Occupancy")
        ax.legend()
        ax.grid(alpha=0.2)

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.info(
        "This dashboard displays results from the Mall Dataset. "
        "It is not connected to a live CCTV camera."
    )


# ==================================================
# PAGE 2: DETECTION RESULTS
# ==================================================

elif page == "Detection Results":

    st.subheader("🤖 YOLOv8 Person Detection")

    st.write(
        "The YOLOv8 model identifies people in mall images "
        "and estimates the number of detected persons."
    )

    if sample_image.exists():

        with st.container(border=True):

            st.markdown("#### Sample Detection Image")

            st.image(
                str(sample_image),
                caption="YOLOv8 Person Detection Result",
                width="stretch"
            )

    else:
        st.warning("Detection image not found.")

    st.divider()

    st.markdown("#### Detection Summary")

    col1, col2 = st.columns(2)

    col1.metric("Average People Detected", f"{detected.mean():.2f}")
    col2.metric("Maximum People Detected", int(detected.max()))

    st.markdown("#### Detection Performance")

    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

    st.warning(
        "The model undercounts people in many frames. "
        "This may happen when people are small, overlapping "
        "or partially hidden."
    )


# ==================================================
# PAGE 3: FRAME ANALYSIS
# ==================================================

elif page == "Frame Analysis":

    st.subheader("🔍 Frame-by-Frame Analysis")

    st.write(
        "Explore the actual and YOLO-detected people counts "
        "for individual frames."
    )

    frame_search = st.text_input(
        "Search frame name",
        placeholder="Example: seq_000001.jpg"
    )

    filtered_data = data.copy()

    if frame_search:
        filtered_data = filtered_data[
            filtered_data["Frame"].astype(str).str.contains(
                frame_search,
                case=False,
                na=False
            )
        ].copy()

    filtered_data["Detection Error"] = (
        filtered_data["Actual Count"] -
        filtered_data["Detected Count"]
    ).abs()

    st.markdown("#### Frame Results")

    st.dataframe(
        filtered_data,
        width="stretch",
        hide_index=True
    )

    st.caption(f"Showing {len(filtered_data)} frame records.")

    st.divider()

    if not filtered_data.empty:

        st.markdown("#### Selected Frame Occupancy")

        selected_frame = st.selectbox(
            "Choose a frame",
            filtered_data["Frame"].astype(str).tolist()
        )

        selected_row = filtered_data[
            filtered_data["Frame"].astype(str) == selected_frame
        ].iloc[0]

        st.info(
            f"Selected Dataset Frame: {selected_frame}. "
            "This is an image-based estimate, not live CCTV occupancy."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Actual Occupancy", int(selected_row["Actual Count"]))
        col2.metric("YOLO Occupancy Estimate", int(selected_row["Detected Count"]))
        col3.metric("Detection Error", int(selected_row["Detection Error"]))

    else:
        st.info("No matching frames found.")


# ==================================================
# PAGE 4: UPLOAD AND DETECT
# ==================================================

elif page == "Upload & Detect":

    st.subheader("📤 Upload Image or Video")

    st.write(
        "Upload your own image or video to detect people "
        "using the YOLOv8n model."
    )

    upload_type = st.radio(
        "Choose upload type",
        ["Image", "Video"],
        horizontal=True
    )

    # ---------------- IMAGE UPLOAD ----------------

    if upload_type == "Image":

        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )

        if uploaded_image is not None:

            image = Image.open(uploaded_image).convert("RGB")

            st.markdown("#### Uploaded Image")
            st.image(image, width="stretch")

            if st.button("Detect People in Image"):

                with st.spinner("Loading YOLOv8n and detecting people..."):

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

                col1, col2 = st.columns(2)

                col1.metric("People Detected", person_count)
                col2.metric("Model", "YOLOv8n")

                st.markdown("#### Detection Result")

                st.image(
                    annotated_image,
                    caption="Detected People with Bounding Boxes",
                    width="stretch"
                )

                image_bytes = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                )[1].tobytes()

                st.download_button(
                    "Download Detection Image",
                    data=image_bytes,
                    file_name="mall_detection_result.jpg",
                    mime="image/jpeg"
                )

    # ---------------- VIDEO UPLOAD ----------------

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
                help="1 means every frame is processed. Higher values process fewer frames."
            )

            if st.button("Detect People in Video"):

                model = load_model()

                input_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
                input_temp.write(uploaded_video.getbuffer())
                input_temp.close()

                output_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
                output_path = output_temp.name
                output_temp.close()

                cap = cv2.VideoCapture(input_temp.name)

                if not cap.isOpened():

                    st.error("Unable to read this video file.")

                else:

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 25.0

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps / frame_interval,
                        (width, height)
                    )

                    if not writer.isOpened():

                        cap.release()
                        st.error("Unable to create the processed video.")

                    else:

                        progress = st.progress(0)
                        status = st.empty()

                        frame_number = 0
                        processed_number = 0
                        counts = []

                        with st.spinner("Processing video frames..."):

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

                                    processed_number += 1

                                frame_number += 1

                                if total_frames > 0:
                                    progress.progress(
                                        min(frame_number / total_frames, 1.0)
                                    )

                                if frame_number % 30 == 0:
                                    status.write(
                                        f"Processed {frame_number} of {total_frames} frames"
                                    )

                        cap.release()
                        writer.release()

                        progress.progress(1.0)
                        status.success("Video processing completed!")

                        if counts:

                            results_df = pd.DataFrame({
                                "Processed Frame": range(1, len(counts) + 1),
                                "Detected People": counts
                            })

                            st.success("Video detection completed!")

                            col1, col2, col3 = st.columns(3)

                            col1.metric("Processed Frames", len(counts))
                            col2.metric("Average People", f"{np.mean(counts):.2f}")
                            col3.metric("Peak People", int(max(counts)))

                            st.markdown("#### Processed Video")

                            with open(output_path, "rb") as video_file:
                                video_bytes = video_file.read()

                            st.video(video_bytes)

                            st.download_button(
                                "Download Processed Video",
                                data=video_bytes,
                                file_name="mall_processed_video.mp4",
                                mime="video/mp4"
                            )

                            st.markdown("#### Frame-wise People Count")

                            st.dataframe(
                                results_df,
                                width="stretch",
                                hide_index=True
                            )

                            st.markdown("#### Video Occupancy Trend")

                            st.line_chart(
                                results_df.set_index("Processed Frame")
                            )

                            csv_bytes = results_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                "Download Frame Count CSV",
                                data=csv_bytes,
                                file_name="video_occupancy_results.csv",
                                mime="text/csv"
                            )

                        else:
                            st.warning("No frames could be processed.")


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<div style='text-align:center; color:#648078; padding:10px;'>"
    "MallSight AI | AI-Based Mall Occupancy Monitoring System"
    "<br>Powered by Python, Streamlit and YOLOv8"
    "</div>",
    unsafe_allow_html=True
)
