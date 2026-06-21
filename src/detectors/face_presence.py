import os
import time
import yaml
import logging
import numpy as np
from src.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


class FacePresenceDetector:
    """MediaPipe Face Mesh-based detector to check face presence and detect multi-face events.

    Known limitations:
        - Photos, posters, or screens displaying faces in the background can cause false-positive
          multi-face detection.
        - Low light, heavy backlighting, or extreme head rotation can prevent face detection,
          leading to false "no face" triggers.
        - Motion blur from rapid movement can temporarily drop face detection.
        - This detector does not perform facial recognition or verification, only presence counting.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml") -> None:
        """Initializes the FacePresenceDetector, loading thresholds and initializing Face Mesh.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.absence_duration_sec: float = 1.5

        # Load thresholds config
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config and "face_presence" in config:
                    fp_cfg = config["face_presence"]
                    self.absence_duration_sec = float(fp_cfg.get("absence_duration_sec", 1.5))
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}. Using defaults.")
        else:
            logger.warning(f"Config file not found at {config_path}. Using defaults.")

        # Initialize MediaPipe Face Mesh
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            # Distinct FaceMesh instance with max_num_faces=3
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                refine_landmarks=False,
                max_num_faces=3,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Face Mesh: {e}")
            raise

        self.first_absence_time: float | None = None

    def analyze(self, frame: np.ndarray, face_mesh_results = None) -> DetectionResult:
        """Analyzes a video frame to check face presence and count faces.

        Args:
            frame (np.ndarray): Input video frame in BGR format.
            face_mesh_results (Optional[Any]): Pre-processed Face Mesh results.

        Returns:
            DetectionResult: The face presence detection results.
        """
        now = time.time()
        h, w = frame.shape[:2]

        if face_mesh_results is not None:
            results = face_mesh_results
        else:
            # Convert frame to RGB for MediaPipe
            try:
                rgb_frame = cv2_cvtColor_safe(frame)
                results = self.face_mesh.process(rgb_frame)
            except Exception as e:
                logger.error(f"MediaPipe Face Mesh processing failed: {e}")
                return self._create_error_result(now, str(e))

        face_count = 0
        boxes = []
        confidences = []

        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)
            for face in results.multi_face_landmarks:
                # Calculate bounding box from landmarks
                xs = [lm.x for lm in face.landmark]
                ys = [lm.y for lm in face.landmark]
                xmin = float(min(xs) * w)
                ymin = float(min(ys) * h)
                xmax = float(max(xs) * w)
                ymax = float(max(ys) * h)
                boxes.append([xmin, ymin, xmax, ymax])

                # Retrieve confidence if present (e.g. from mocks), else default to 0.90
                conf = 0.90
                face_confidence = getattr(face, "confidence", None)
                face_score = getattr(face, "score", None)
                if face_confidence is not None and isinstance(face_confidence, (int, float)):
                    conf = float(face_confidence)
                elif face_score is not None and isinstance(face_score, (int, float)):
                    conf = float(face_score)
                conf = min(conf, 0.99)
                confidences.append(conf)

        # Trigger logic
        triggered = False
        confidence = 0.0

        if face_count == 0:
            if self.first_absence_time is None:
                self.first_absence_time = now
            absence_duration = now - self.first_absence_time
            if absence_duration >= self.absence_duration_sec:
                triggered = True
            confidence = 0.0
        elif face_count == 1:
            self.first_absence_time = None
            triggered = False
            confidence = confidences[0] if confidences else 0.90
        else:  # face_count > 1
            self.first_absence_time = None
            triggered = True
            confidence = min(confidences) if confidences else 0.90

        details = {
            "face_count": face_count,
            "boxes": boxes,
            "absence_duration": float(now - self.first_absence_time)
            if self.first_absence_time is not None
            else 0.0,
        }

        return DetectionResult(
            signal_name="face_presence",
            triggered=triggered,
            confidence=confidence,
            details=details,
            timestamp=now,
        )

    def _create_error_result(self, timestamp: float, error_msg: str) -> DetectionResult:
        """Helper to create a default result on error."""
        details = {
            "face_count": 0,
            "boxes": [],
            "absence_duration": 0.0,
            "error": error_msg,
        }
        return DetectionResult(
            signal_name="face_presence",
            triggered=False,
            confidence=0.0,
            details=details,
            timestamp=timestamp,
        )


def cv2_cvtColor_safe(frame: np.ndarray) -> np.ndarray:
    """Safe wrapper around cv2.cvtColor to avoid unresolved import issues."""
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
