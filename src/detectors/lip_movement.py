import os
import time
import yaml
import logging
import numpy as np
from typing import List, Tuple
from src.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


class LipMovementDetector:
    """MediaPipe Face Mesh-based detector for estimating lip movement to detect talking.

    Known limitations:
        - Occlusions from hands, facial hair, masks, or microphones will reduce accuracy.
        - Backlighting, low-light conditions, or motion blur can cause landmark jitter, impacting MAR variance.
        - Squinting, chewing gum, yawning, or exaggerated facial expressions may produce false positives.
        - Cannot differentiate between talking-to-self, reading aloud, and talking to someone else.
        - This is a motion-pattern detection signal, not speech or voice detection.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml") -> None:
        """Initializes the LipMovementDetector, loading config thresholds and initializing Face Mesh.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Default fallback values
        self.movement_var_threshold: float = 0.02
        self.buffer_duration_sec: float = 1.5
        self.min_duration_sec: float = 0.5
        self.min_crossings: int = 3

        # Load config
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config and "lip" in config:
                    lip_cfg = config["lip"]
                    self.movement_var_threshold = float(lip_cfg.get("movement_var_threshold", 0.02))
                    self.buffer_duration_sec = float(lip_cfg.get("buffer_duration_sec", 1.5))
                    self.min_duration_sec = float(lip_cfg.get("min_duration_sec", 0.5))
                    self.min_crossings = int(lip_cfg.get("min_crossings", 3))
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

        # Rolling buffer for MAR: list of tuples (timestamp, mar_value)
        self.mar_buffer: List[Tuple[float, float]] = []

    def analyze(self, frame: np.ndarray, face_mesh_results = None) -> DetectionResult:
        """Analyzes a video frame to calculate MAR and determine if the user is talking.

        Args:
            frame (np.ndarray): Input video frame in BGR format.
            face_mesh_results (Optional[Any]): Pre-processed Face Mesh results.

        Returns:
            DetectionResult: The lip movement detection results.
        """
        now = time.time()

        if face_mesh_results is not None:
            results = face_mesh_results
        else:
            # Convert frame to RGB for MediaPipe Face Mesh
            try:
                rgb_frame = cv2_cvtColor_safe(frame)
                results = self.face_mesh.process(rgb_frame)
            except Exception as e:
                logger.error(f"MediaPipe Face Mesh processing failed: {e}")
                return self._create_no_face_result(now, error_msg=str(e))

        # Reset buffer if face is lost
        if not results.multi_face_landmarks:
            self.mar_buffer.clear()
            return self._create_no_face_result(now)

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        try:
            # Extract relevant inner lip landmarks:
            # - Inner upper lip center: 13
            # - Inner lower lip center: 14
            # - Inner subject's right mouth corner (left of image): 78
            # - Inner subject's left mouth corner (right of image): 308
            lm13 = landmarks[13]
            lm14 = landmarks[14]
            lm78 = landmarks[78]
            lm308 = landmarks[308]

            # Scale to pixel coordinates
            p13 = np.array([lm13.x * w, lm13.y * h])
            p14 = np.array([lm14.x * w, lm14.y * h])
            p78 = np.array([lm78.x * w, lm78.y * h])
            p308 = np.array([lm308.x * w, lm308.y * h])

            # Calculate distances
            vertical_dist = np.linalg.norm(p13 - p14)
            horizontal_dist = np.linalg.norm(p78 - p308)

            # Calculate Mouth Aspect Ratio (MAR)
            if horizontal_dist > 1e-5:
                mar = float(vertical_dist / horizontal_dist)
            else:
                mar = 0.0

            # Update rolling buffer
            self.mar_buffer.append((now, mar))

            # Filter old samples from buffer
            cutoff_time = now - self.buffer_duration_sec
            self.mar_buffer = [item for item in self.mar_buffer if item[0] >= cutoff_time]

            # Determine stats and trigger
            triggered = False
            confidence = 0.0
            mar_variance = 0.0
            mar_mean = 0.0
            buffer_time_span = 0.0
            crossings = 0

            if len(self.mar_buffer) > 1:
                buffer_time_span = self.mar_buffer[-1][0] - self.mar_buffer[0][0]

                # Compute variance if time span is sufficient
                if buffer_time_span >= self.min_duration_sec:
                    mar_values = [item[1] for item in self.mar_buffer]
                    mar_variance = float(np.var(mar_values))
                    mar_mean = float(np.mean(mar_values))

                    # Count crossings of the mean MAR to detect oscillation
                    margin = 0.02
                    state = None  # True for above mean, False for below mean
                    for val in mar_values:
                        if state is None:
                            if val > mar_mean + margin:
                                state = True
                            elif val < mar_mean - margin:
                                state = False
                        else:
                            if state and val < mar_mean - margin:
                                state = False
                                crossings += 1
                            elif not state and val > mar_mean + margin:
                                state = True
                                crossings += 1

                    if mar_variance > self.movement_var_threshold and crossings >= self.min_crossings:
                        triggered = True
                        # Confidence score scales from 0.5 up to 0.99
                        # based on the magnitude of the variance past the threshold.
                        # Using 0.05 as a normalization factor for the excess variance.
                        excess = mar_variance - self.movement_var_threshold
                        confidence = float(min(0.99, 0.5 + excess / 0.05))

            details = {
                "mar_value": mar,
                "mar_mean": mar_mean,
                "mar_variance": mar_variance,
                "buffer_len": len(self.mar_buffer),
                "buffer_duration": buffer_time_span,
                "crossings": crossings,
            }

            return DetectionResult(
                signal_name="lip_movement",
                triggered=triggered,
                confidence=confidence,
                details=details,
                timestamp=now,
            )

        except Exception as e:
            logger.error(f"Error calculating lip movement: {e}")
            return self._create_no_face_result(now, error_msg=f"MAR calculation error: {e}")

    def _create_no_face_result(self, timestamp: float, error_msg: str = "") -> DetectionResult:
        """Helper to create a default result when no face is detected or on errors."""
        details = {
            "mar_value": 0.0,
            "mar_mean": 0.0,
            "mar_variance": 0.0,
            "buffer_len": 0,
            "buffer_duration": 0.0,
        }
        if error_msg:
            details["error"] = error_msg
        else:
            details["error"] = "No face detected"

        return DetectionResult(
            signal_name="lip_movement",
            triggered=False,
            confidence=0.0,
            details=details,
            timestamp=timestamp,
        )


def cv2_cvtColor_safe(frame: np.ndarray) -> np.ndarray:
    """Safe wrapper around cv2.cvtColor to avoid unresolved import issues."""
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
