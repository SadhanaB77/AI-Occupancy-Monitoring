
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

/* Main background */
.stApp {
    background-color: #F5FAF9;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #E6F4F1;
    border-right: 1px solid #D4E8E3;
}

/* Headings */
h1, h2, h3 {
    color: #164E45 !important;
}

/* Metric cards */
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

/* Section containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border-radius: 14px;
}

/* Buttons */
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

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #DCEBE7;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------

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
            "Frame Analysis"
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

    # Actual occupancy cards
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

    # Occupancy chart
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
            linewidth=1.5,
            alpha=0.9
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
        "Note: The dashboard displays results from the Mall Dataset. "
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
        "This may happen when people are small, overlapping or partially hidden."
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

    # Search/filter frames
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
        ]

    # Show error column
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

        st.markdown("#### Selected Frame Statistics")

        selected_frame = st.selectbox(
            "Choose a frame",
            filtered_data["Frame"].astype(str).tolist()
        )

        selected_row = filtered_data[
            filtered_data["Frame"].astype(str) == selected_frame
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Actual People",
            int(selected_row["Actual Count"])
        )

        col2.metric(
            "YOLO Detected",
            int(selected_row["Detected Count"])
        )

        col3.metric(
            "Detection Error",
            int(selected_row["Detection Error"])
        )

    else:
        st.info("No matching frames found.")


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<div style='text-align:center; color:#648078; padding:10px;'>"
    "MallSight AI | AI-Based Mall Occupancy Monitoring System"
    "<br>Powered by Python, Streamlit and YOLOv8"
    "</div>",
    unsafe_allow_html=True
)
