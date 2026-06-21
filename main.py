import collections
import logging
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

from src.alert_engine import AlertEngine
from src.capture import CameraNotFoundError, VideoCapture
from src.detectors.base import DetectionResult
from src.event_logger import EventLogger
from src.pipeline import ProctorPipeline

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def draw_translucent_panel(
    img: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int] = (15, 15, 15), alpha: float = 0.7
) -> None:
    """Draws a translucent panel to act as a HUD/UI background.

    Args:
        img (np.ndarray): Image to draw onto.
        x (int): Top-left X coordinate.
        y (int): Top-left Y coordinate.
        w (int): Panel width.
        h (int): Panel height.
        color (Tuple[int, int, int]): Panel background color (BGR).
        alpha (float): Opacity of the background panel.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_face_bounding_box(img: np.ndarray, box: List[float], color: Tuple[int, int, int], text: str) -> None:
    """Draws a futuristic bounding box for faces with corner tick marks.

    Args:
        img (np.ndarray): Image to draw onto.
        box (List[float]): Bounding box [xmin, ymin, xmax, ymax].
        color (Tuple[int, int, int]): Box and text color (BGR).
        text (str): Label to display.
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    # Draw thin main bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Draw corner tick marks (thicker lines)
    tick_len = min(15, int((x2 - x1) * 0.15))
    tick_thick = 2

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + tick_len, y1), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + tick_len), color, tick_thick, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - tick_len, y1), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + tick_len), color, tick_thick, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + tick_len, y2), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - tick_len), color, tick_thick, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - tick_len, y2), color, tick_thick, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - tick_len), color, tick_thick, cv2.LINE_AA)

    # Label text background
    label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)[0]
    ly2 = max(y1 - 4, label_size[1] + 4)
    cv2.rectangle(img, (x1, ly2 - label_size[1] - 4), (x1 + label_size[0] + 6, ly2 + 2), (15, 15, 15), -1)
    cv2.rectangle(img, (x1, ly2 - label_size[1] - 4), (x1 + label_size[0] + 6, ly2 + 2), color, 1)
    cv2.putText(img, text, (x1 + 3, ly2 - 1), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_object_bounding_box(img: np.ndarray, box: List[float], color: Tuple[int, int, int], text: str) -> None:
    """Draws a clean bounding box for detected objects.

    Args:
        img (np.ndarray): Image to draw onto.
        box (List[float]): Bounding box [xmin, ymin, xmax, ymax].
        color (Tuple[int, int, int]): Box color (BGR).
        text (str): Label to display.
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    # Bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label background
    label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)[0]
    ly2 = max(y1 - 4, label_size[1] + 4)
    cv2.rectangle(img, (x1, ly2 - label_size[1] - 4), (x1 + label_size[0] + 6, ly2 + 2), color, -1)
    cv2.putText(img, text, (x1 + 3, ly2 - 1), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    """Main application loop orchestrating the video capture and AI proctoring pipeline."""
    # 1. Initialize Video Capture (high-res full display)
    try:
        cap = VideoCapture(index=0, max_width=1280)
    except CameraNotFoundError as e:
        logger.critical(f"Camera Initialization Failed: {e}")
        sys.exit(1)

    # 2. Initialize Logging and Proctoring Core
    event_logger = EventLogger(log_dir="logs", batch_size=5)
    alert_engine = AlertEngine(config_path="config/thresholds.yaml", event_logger=event_logger)
    pipeline = ProctorPipeline(config_path="config/thresholds.yaml", model_path="models/yolov8n.pt")

    logger.info("Proctoring pipeline successfully loaded. Starting monitoring...")

    # Performance logging
    fps_history = collections.deque(maxlen=30)
    frame_id = 0

    try:
        while True:
            t_start = time.perf_counter()

            # Read frame
            raw_frame = cap.read_frame()
            if raw_frame is None:
                logger.error("Failed to read frame from camera.")
                break

            h, w = raw_frame.shape[:2]

            # 3. Downscale frame to max 640px width for inference (performance budget rule)
            inference_width = 640
            if w > inference_width:
                scale = float(inference_width) / w
                inference_frame = cv2.resize(raw_frame, (inference_width, int(h * scale)))
            else:
                scale = 1.0
                inference_frame = raw_frame.copy()

            inf_h, inf_w = inference_frame.shape[:2]
            scale_x = w / float(inf_w)
            scale_y = h / float(inf_h)

            # 4. Run the Pipeline and collect alerts
            results = pipeline.process_frame(inference_frame)
            active_alerts = alert_engine.process(results, frame_id=frame_id)

            # 5. Measure loop FPS
            t_end = time.perf_counter()
            loop_duration = t_end - t_start
            if loop_duration > 0:
                fps_history.append(1.0 / loop_duration)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            # 6. Render HUD and Overlays

            # Draw Status HUD Panel (Top-Left)
            hud_x, hud_y = 20, 20
            draw_translucent_panel(raw_frame, hud_x, hud_y, 230, 70, (15, 15, 15), 0.7)
            # Solid vertical accent bar (green status theme)
            cv2.rectangle(raw_frame, (hud_x, hud_y), (hud_x + 6, hud_y + 70), (100, 220, 100), -1)

            # Pulsing status dot
            dot_center = (hud_x + 25, hud_y + 25)
            pulse_radius = int(5.5 + 1.5 * np.sin(time.time() * 5))
            cv2.circle(raw_frame, dot_center, pulse_radius, (100, 220, 100), -1)

            cv2.putText(
                raw_frame,
                "MONITORING",
                (hud_x + 42, hud_y + 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                raw_frame,
                f"Pipeline Speed: {avg_fps:.1f} FPS",
                (hud_x + 20, hud_y + 53),
                cv2.FONT_HERSHEY_DUPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Draw Real-time Telemetry console (Bottom-Left)
            tel_x, tel_y = 20, h - 130
            draw_translucent_panel(raw_frame, tel_x, tel_y, 250, 110, (15, 15, 15), 0.6)
            cv2.rectangle(raw_frame, (tel_x, tel_y), (tel_x + 6, tel_y + 110), (180, 180, 180), -1)

            # Extract telemetry values
            gaze_angle, head_yaw, head_pitch, shoulder_shift, lip_mar, lip_var = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            face_count = 0
            for res in results:
                if res.signal_name == "eye_gaze":
                    gaze_angle = res.details.get("gaze_angle_deg", 0.0)
                elif res.signal_name == "head_pose":
                    head_yaw = res.details.get("yaw_deg", 0.0)
                    head_pitch = res.details.get("pitch_deg", 0.0)
                elif res.signal_name == "shoulder_pose":
                    shoulder_shift = res.details.get("lateral_shift_ratio", 0.0)
                elif res.signal_name == "lip_movement":
                    lip_mar = res.details.get("mar_value", 0.0)
                    lip_var = res.details.get("mar_variance", 0.0)
                elif res.signal_name == "face_presence":
                    face_count = res.details.get("face_count", 0)

            # Print telemetry strings
            cv2.putText(
                raw_frame,
                "REAL-TIME DIAGNOSTICS",
                (tel_x + 18, tel_y + 20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                raw_frame,
                f"Gaze Angle: {gaze_angle:.1f} deg",
                (tel_x + 18, tel_y + 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.35,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                raw_frame,
                f"Head Pose: Yaw: {head_yaw:.1f} Pitch: {head_pitch:.1f}",
                (tel_x + 18, tel_y + 55),
                cv2.FONT_HERSHEY_DUPLEX,
                0.35,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                raw_frame,
                f"Shoulder Posture Shift: {shoulder_shift:.3f}",
                (tel_x + 18, tel_y + 70),
                cv2.FONT_HERSHEY_DUPLEX,
                0.35,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                raw_frame,
                f"Mouth MAR: {lip_mar:.2f} Variance: {lip_var:.4f}",
                (tel_x + 18, tel_y + 85),
                cv2.FONT_HERSHEY_DUPLEX,
                0.35,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # Draw Bounding Boxes (Scaled up to high-resolution display coordinates)
            for res in results:
                # 1. Faces (from face presence check)
                if res.signal_name == "face_presence":
                    boxes = res.details.get("boxes", [])
                    for box in boxes:
                        rx1, ry1, rx2, ry2 = (
                            box[0] * scale_x,
                            box[1] * scale_y,
                            box[2] * scale_x,
                            box[3] * scale_y,
                        )
                        # Red box if face presence triggered (multi-face or missing), otherwise gold/cyan
                        color = (80, 80, 240) if res.triggered else (220, 180, 100)
                        lbl = f"Face: Conf {res.confidence:.2f}" if not res.triggered else "FACE PRESENCE WARNING"
                        draw_face_bounding_box(raw_frame, [rx1, ry1, rx2, ry2], color, lbl)

                # 2. Objects (from YOLO)
                elif res.signal_name == "person_count":
                    boxes = res.details.get("boxes", [])
                    for box in boxes:
                        rx1, ry1, rx2, ry2 = (
                            box[0] * scale_x,
                            box[1] * scale_y,
                            box[2] * scale_x,
                            box[3] * scale_y,
                        )
                        color = (0, 140, 255) if res.triggered else (180, 220, 100)
                        lbl = f"Person (Conf {res.confidence:.2f})"
                        draw_object_bounding_box(raw_frame, [rx1, ry1, rx2, ry2], color, lbl)

                elif res.signal_name == "phone_detected":
                    boxes = res.details.get("boxes", [])
                    for box in boxes:
                        rx1, ry1, rx2, ry2 = (
                            box[0] * scale_x,
                            box[1] * scale_y,
                            box[2] * scale_x,
                            box[3] * scale_y,
                        )
                        color = (80, 80, 240)  # Crimson Alert
                        lbl = f"MOBILE PHONE: Conf {res.confidence:.2f}"
                        draw_object_bounding_box(raw_frame, [rx1, ry1, rx2, ry2], color, lbl)

                elif res.signal_name == "secondary_device_detected":
                    # Laptops (since cell phones are drawn under phone_detected)
                    boxes = res.details.get("boxes", [])
                    phone_cnt = res.details.get("phone_count", 0)
                    # Laptop boxes are appended after phone boxes in details["boxes"]
                    laptop_boxes = boxes[phone_cnt:]
                    for box in laptop_boxes:
                        rx1, ry1, rx2, ry2 = (
                            box[0] * scale_x,
                            box[1] * scale_y,
                            box[2] * scale_x,
                            box[3] * scale_y,
                        )
                        color = (0, 140, 255)  # Orange Warn
                        lbl = f"Secondary Screen (Conf {res.confidence:.2f})"
                        draw_object_bounding_box(raw_frame, [rx1, ry1, rx2, ry2], color, lbl)

            # Draw Alert Engine Notification Stack (Top-Right)
            alert_w = 340
            alert_h = 45
            alert_gap = 10
            x_alert = w - alert_w - 20
            y_alert = 20

            for alert_msg in active_alerts:
                # Color code based on severity
                if " - HIGH" in alert_msg:
                    sev_color = (80, 80, 240)  # Crimson/Coral
                elif " - MEDIUM" in alert_msg:
                    sev_color = (0, 140, 255)  # Warm Amber/Orange
                else:
                    sev_color = (128, 255, 255)  # Soft Gold/Yellow

                # Clean message of the severity suffix for presentation
                clean_msg = alert_msg
                for suffix in [" - HIGH", " - MEDIUM", " - LOW"]:
                    if clean_msg.endswith(suffix):
                        clean_msg = clean_msg[: -len(suffix)]
                        break

                # Draw notification card background
                draw_translucent_panel(raw_frame, x_alert, y_alert, alert_w, alert_h, (15, 15, 15), 0.75)
                # Left accent indicator bar showing severity color
                cv2.rectangle(raw_frame, (x_alert, y_alert), (x_alert + 8, y_alert + alert_h), sev_color, -1)

                # Draw message text
                cv2.putText(
                    raw_frame,
                    clean_msg,
                    (x_alert + 18, y_alert + 26),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.38,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

                y_alert += alert_h + alert_gap

            # Render display
            cv2.imshow("Proctoring System - Real-time AI Proctor", raw_frame)

            frame_id += 1

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User requested exit.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        # Graceful cleanup
        cap.release()
        event_logger.close()
        cv2.destroyAllWindows()
        logger.info("Proctoring session ended. Resources released and log buffers flushed.")


if __name__ == "__main__":
    main()
