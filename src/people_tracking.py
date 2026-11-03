
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


def _open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    return cap, width, height, fps


def track_line_crossing(
    video_path,
    output_path="reports/line_crossing_output.mp4",
    line_position=0.5,
):
    """Track people and count crossings in both directions."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cap, width, height, fps = _open_video(video_path)
    line_y = int(height * line_position)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the processed video.")

    model = YOLO("yolov8n.pt")

    previous_positions = {}
    counted_down = set()
    counted_up = set()

    top_to_bottom = 0
    bottom_to_top = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            result = model.track(
                frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                conf=0.25,
                verbose=False,
            )[0]

            cv2.line(
                frame,
                (0, line_y),
                (width, line_y),
                (255, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                "Virtual Counting Line",
                (10, max(line_y - 12, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 0),
                2,
            )

            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                coordinates = boxes.xyxy.cpu().numpy()

                for person_id, box in zip(ids, coordinates):
                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    if person_id in previous_positions:
                        previous_y = previous_positions[person_id]

                        # Top to bottom
                        if (
                            previous_y < line_y <= center_y
                            and person_id not in counted_down
                        ):
                            top_to_bottom += 1
                            counted_down.add(person_id)

                        # Bottom to top
                        elif (
                            previous_y > line_y >= center_y
                            and person_id not in counted_up
                        ):
                            bottom_to_top += 1
                            counted_up.add(person_id)

                    previous_positions[person_id] = center_y

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 200, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"ID {person_id}",
                        (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 200, 0),
                        2,
                    )

            cv2.putText(
                frame,
                f"Top to Bottom: {top_to_bottom}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Bottom to Top: {bottom_to_top}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    summary = pd.DataFrame({
        "Direction": ["Top to Bottom", "Bottom to Top"],
        "People Count": [top_to_bottom, bottom_to_top],
    })

    summary.to_csv(
        "reports/line_crossing_summary.csv",
        index=False,
    )

    return {
        "top_to_bottom": top_to_bottom,
        "bottom_to_top": bottom_to_top,
        "output_path": str(output_path),
        "summary": summary,
    }


def count_people_in_region(
    video_path,
    roi=(0.25, 0.25, 0.75, 0.75),
    output_path="reports/area_counting_output.mp4",
):
    """Count tracked people whose centre is inside the selected rectangle."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cap, width, height, fps = _open_video(video_path)

    x1_ratio, y1_ratio, x2_ratio, y2_ratio = roi

    x1_roi = int(width * x1_ratio)
    y1_roi = int(height * y1_ratio)
    x2_roi = int(width * x2_ratio)
    y2_roi = int(height * y2_ratio)

    if x1_roi >= x2_roi or y1_roi >= y2_roi:
        cap.release()
        raise ValueError("Invalid selected area. Check the ROI coordinates.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the processed video.")

    model = YOLO("yolov8n.pt")
    frame_counts = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            result = model.track(
                frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                conf=0.25,
                verbose=False,
            )[0]

            inside_count = 0
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                coordinates = boxes.xyxy.cpu().numpy()

                for person_id, box in zip(ids, coordinates):
                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    inside = (
                        x1_roi <= center_x <= x2_roi
                        and y1_roi <= center_y <= y2_roi
                    )

                    if inside:
                        inside_count += 1
                        box_color = (0, 200, 0)
                    else:
                        box_color = (130, 130, 130)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2,
                    )

                    label = f"ID {person_id}"

                    if inside:
                        label += " - IN AREA"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )

            frame_counts.append(inside_count)

            cv2.rectangle(
                frame,
                (x1_roi, y1_roi),
                (x2_roi, y2_roi),
                (255, 0, 0),
                3,
            )

            cv2.putText(
                frame,
                f"People in selected area: {inside_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    counts_df = pd.DataFrame({
        "Processed Frame": range(1, len(frame_counts) + 1),
        "People Inside Selected Area": frame_counts,
    })

    counts_df.to_csv(
        "reports/area_counting_results.csv",
        index=False,
    )

    return {
        "frame_counts": frame_counts,
        "average": float(pd.Series(frame_counts).mean()) if frame_counts else 0.0,
        "peak": int(max(frame_counts)) if frame_counts else 0,
        "last_frame_count": int(frame_counts[-1]) if frame_counts else 0,
        "output_path": str(output_path),
        "results": counts_df,
    }
