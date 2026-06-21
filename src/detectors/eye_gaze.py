import os
import time
import yaml
import logging
import numpy as np
from src.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


class EyeGazeDetector:
    """MediaPipe Face Mesh-based detector for estimating eye gaze direction and looking away events.

    Known limitations:
        - Accuracy drops with glasses glare, extreme or uneven lighting, and backlighting.
        - Accuracy decreases if the subject is squinting, has eyes partially closed, or has
          extreme head rotation (side profile) which occludes one or both eyes.
        - Simple ratio-based gaze estimation does not account for user-specific eyeball shape/size
          without individual calibration.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml") -> None:
        """Initializes the EyeGazeDetector, loading config thresholds and initializing Face Mesh.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Default fallback values
        self.away_angle_deg: float = 25.0
        self.away_duration_sec: float = 2.0

        # Sensitivity factors to map gaze ratio deviation from 0.5 to estimated degrees
        self.sensitivity_h: float = 80.0
        self.sensitivity_v: float = 80.0

        # Load config
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config and "eye_gaze" in config:
                    eg_cfg = config["eye_gaze"]
                    self.away_angle_deg = float(eg_cfg.get("away_angle_deg", 25.0))
                    self.away_duration_sec = float(eg_cfg.get("away_duration_sec", 2.0))
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}. Using defaults.")
        else:
            logger.warning(f"Config file not found at {config_path}. Using defaults.")

        # Initialize MediaPipe Face Mesh
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Face Mesh: {e}")
            raise

        # Gaze tracking state
        self.first_look_away_time: float | None = None

    def analyze(self, frame: np.ndarray, face_mesh_results = None) -> DetectionResult:
        """Analyzes a video frame to estimate gaze direction and detect looking away.

        Args:
            frame (np.ndarray): Input video frame in BGR format.
            face_mesh_results (Optional[Any]): Pre-processed Face Mesh results.

        Returns:
            DetectionResult: The gaze detection results.
        """
        now = time.time()

        if face_mesh_results is not None:
            results = face_mesh_results
        else:
            # Convert frame to RGB for MediaPipe
            try:
                rgb_frame = cv2_cvtColor_safe(frame)
                results = self.face_mesh.process(rgb_frame)
            except Exception as e:
                logger.error(f"MediaPipe Face Mesh processing failed: {e}")
                return self._create_no_face_result(now, error_msg=str(e))

        # Check if face landmarks are found
        if not results.multi_face_landmarks:
            # Reset timer if face is lost
            self.first_look_away_time = None
            return self._create_no_face_result(now)

        landmarks = results.multi_face_landmarks[0].landmark

        try:
            # Right Eye Landmarks (subject's right, left side of image)
            # Outer corner: 33, Inner corner: 133, Top eyelid: 159, Bottom eyelid: 145, Iris: 468
            re_outer = landmarks[33]
            re_inner = landmarks[133]
            re_top = landmarks[159]
            re_bottom = landmarks[145]
            re_iris = landmarks[468]

            # Left Eye Landmarks (subject's left, right side of image)
            # Outer corner: 263, Inner corner: 362, Top eyelid: 386, Bottom eyelid: 374, Iris: 473
            le_outer = landmarks[263]
            le_inner = landmarks[362]
            le_top = landmarks[386]
            le_bottom = landmarks[374]
            le_iris = landmarks[473]

            # Compute horizontal ratios (0.0 = looking left, 0.5 = center, 1.0 = looking right)
            # For Right Eye: outer.x < inner.x
            hr_denom_r = re_inner.x - re_outer.x
            hr_right = (re_iris.x - re_outer.x) / hr_denom_r if hr_denom_r != 0 else 0.5

            # For Left Eye: inner.x < outer.x
            hr_denom_l = le_outer.x - le_inner.x
            hr_left = (le_iris.x - le_inner.x) / hr_denom_l if hr_denom_l != 0 else 0.5

            # Compute vertical ratios (0.0 = looking up, 0.5 = center, 1.0 = looking down)
            # For Right Eye: top.y < bottom.y
            vr_denom_r = re_bottom.y - re_top.y
            vr_right = (re_iris.y - re_top.y) / vr_denom_r if vr_denom_r != 0 else 0.5

            # For Left Eye: top.y < bottom.y
            vr_denom_l = le_bottom.y - le_top.y
            vr_left = (le_iris.y - le_top.y) / vr_denom_l if vr_denom_l != 0 else 0.5

            # Average ratios across both eyes
            h_ratio = float((hr_right + hr_left) / 2.0)
            v_ratio = float((vr_right + vr_left) / 2.0)

            # Estimate gaze angles relative to straight head pose in degrees
            # (h_ratio - 0.5) > 0 means looking left of frame, < 0 means looking right of frame
            yaw_deg = (h_ratio - 0.5) * self.sensitivity_h
            pitch_deg = (v_ratio - 0.5) * self.sensitivity_v
            gaze_angle_deg = float(np.sqrt(yaw_deg**2 + pitch_deg**2))

            # Determine if gaze angle exceeds threshold
            is_looking_away = gaze_angle_deg >= self.away_angle_deg

            # Maintain rolling timer for trigger state
            triggered = False
            if is_looking_away:
                if self.first_look_away_time is None:
                    self.first_look_away_time = now
                elif now - self.first_look_away_time >= self.away_duration_sec:
                    triggered = True
            else:
                self.first_look_away_time = None

            # Calculate confidence score
            # Confidence is 0.0 if not triggered. If triggered, we combine base level (0.5)
            # with how far past the threshold the angle is, up to 0.99 max.
            # Formula: min(0.99, 0.5 + (gaze_angle_deg - away_angle_deg) / 50.0)
            if triggered:
                confidence = float(min(0.99, 0.5 + (gaze_angle_deg - self.away_angle_deg) / 50.0))
            else:
                confidence = 0.0

            details = {
                "h_ratio": h_ratio,
                "v_ratio": v_ratio,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "gaze_angle_deg": gaze_angle_deg,
                "look_away_duration": float(now - self.first_look_away_time)
                if self.first_look_away_time is not None
                else 0.0,
            }

            return DetectionResult(
                signal_name="eye_gaze",
                triggered=triggered,
                confidence=confidence,
                details=details,
                timestamp=now,
            )

        except Exception as e:
            logger.error(f"Error calculating eye gaze ratios: {e}")
            return self._create_no_face_result(now, error_msg=f"Ratio calculation error: {e}")

    def _create_no_face_result(self, timestamp: float, error_msg: str = "") -> DetectionResult:
        """Helper to create a default result when no face is detected or on errors."""
        details = {"h_ratio": 0.5, "v_ratio": 0.5, "gaze_angle_deg": 0.0, "look_away_duration": 0.0}
        if error_msg:
            details["error"] = error_msg
        else:
            details["error"] = "No face detected"

        return DetectionResult(
            signal_name="eye_gaze",
            triggered=False,
            confidence=0.0,
            details=details,
            timestamp=timestamp,
        )


def cv2_cvtColor_safe(frame: np.ndarray) -> np.ndarray:
    """Safe wrapper around cv2.cvtColor to avoid unresolved import issues."""
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
