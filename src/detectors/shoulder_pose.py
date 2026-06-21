import os
import time
import yaml
import logging
import numpy as np
from src.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


class ShoulderPoseDetector:
    """Estimates shoulder lateral shift and presence using MediaPipe Pose.

    Known limitations:
        - Model complexity 0 trades landmark precision for speed, which is acceptable for coarse shoulder position tracking.
        - Lighting conditions and occlusion (e.g. high-backed chairs or loose clothing) can reduce shoulder detection visibility.
        - Relies on initial baseline setup; extreme positioning at startup might set a skewed baseline.
        - Re-baselining assumes the user is sitting still for 10 seconds, which might absorb a slow, continuous leaning movement if they move extremely slowly.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml") -> None:
        """Initializes the ShoulderPoseDetector, loading config thresholds and initializing Pose.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Default fallback values
        self.lateral_shift_ratio: float = 0.15
        self.min_visibility: float = 0.5
        self.baseline_frames: int = 30
        self.rebaseline_stability_threshold: float = 0.03
        self.rebaseline_duration_sec: float = 10.0

        # Load config
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config and "shoulder" in config:
                    sh_cfg = config["shoulder"]
                    self.lateral_shift_ratio = float(sh_cfg.get("lateral_shift_ratio", 0.15))
                    self.min_visibility = float(sh_cfg.get("min_visibility", 0.5))
                    self.baseline_frames = int(sh_cfg.get("baseline_frames", 30))
                    self.rebaseline_stability_threshold = float(
                        sh_cfg.get("rebaseline_stability_threshold", 0.03)
                    )
                    self.rebaseline_duration_sec = float(
                        sh_cfg.get("rebaseline_duration_sec", 10.0)
                    )
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}. Using defaults.")
        else:
            logger.warning(f"Config file not found at {config_path}. Using defaults.")

        # Initialize MediaPipe Pose
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            # Model complexity 0 is chosen because it runs faster on CPU and provides
            # sufficient accuracy for coarse shoulder position tracking.
            self.pose = self.mp_pose.Pose(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose: {e}")
            raise

        # Detection state
        self.baseline_buffer: list[float] = []
        self.baseline_center_x: float | None = None
        self.stability_buffer: list[tuple[float, float]] = []  # List of (timestamp, center_x)

    def analyze(self, frame: np.ndarray) -> DetectionResult:
        """Analyzes a video frame to estimate shoulder lateral shift and leaning out of frame.

        Args:
            frame (np.ndarray): Input video frame in BGR format.

        Returns:
            DetectionResult: The shoulder pose detection results.
        """
        now = time.time()

        # Convert frame to RGB for MediaPipe
        try:
            rgb_frame = cv2_cvtColor_safe(frame)
            results = self.pose.process(rgb_frame)
        except Exception as e:
            logger.error(f"MediaPipe Pose processing failed: {e}")
            return self._create_no_pose_result(now, error_msg=str(e))

        # Check if pose landmarks are found
        if not results.pose_landmarks:
            self.stability_buffer.clear()
            return self._create_no_pose_result(now)

        landmarks = results.pose_landmarks.landmark

        try:
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            left_vis = float(left_shoulder.visibility)
            right_vis = float(right_shoulder.visibility)

            # Check visibility first. If shoulders are low-visibility, trigger alert.
            if left_vis < self.min_visibility or right_vis < self.min_visibility:
                self.stability_buffer.clear()
                triggered = True
                confidence = float(min(0.99, 0.5 + 0.5 * (1.0 - min(left_vis, right_vis))))

                details = {
                    "left_shoulder_visibility": left_vis,
                    "right_shoulder_visibility": right_vis,
                    "baseline_status": "established" if self.baseline_center_x is not None else "collecting",
                    "error": "Low shoulder visibility (leaning out of frame)"
                }
                if self.baseline_center_x is not None:
                    details["baseline_center_x"] = self.baseline_center_x

                return DetectionResult(
                    signal_name="shoulder_pose",
                    triggered=triggered,
                    confidence=confidence,
                    details=details,
                    timestamp=now
                )

            # Both shoulders are sufficiently visible, calculate center_x
            # MediaPipe's coordinates are normalized to [0.0, 1.0] by frame width
            center_x = float((left_shoulder.x + right_shoulder.x) / 2.0)

            # 1. Establish baseline
            if self.baseline_center_x is None:
                self.baseline_buffer.append(center_x)
                if len(self.baseline_buffer) >= self.baseline_frames:
                    self.baseline_center_x = float(sum(self.baseline_buffer) / len(self.baseline_buffer))
                    logger.info(f"Shoulder pose baseline established: {self.baseline_center_x:.4f}")
                else:
                    # While baseline is being collected, we return triggered=False
                    return DetectionResult(
                        signal_name="shoulder_pose",
                        triggered=False,
                        confidence=0.0,
                        details={
                            "left_shoulder_visibility": left_vis,
                            "right_shoulder_visibility": right_vis,
                            "left_shoulder_x": float(left_shoulder.x),
                            "right_shoulder_x": float(right_shoulder.x),
                            "shoulder_center_x": center_x,
                            "baseline_status": "collecting",
                            "baseline_progress": f"{len(self.baseline_buffer)}/{self.baseline_frames}"
                        },
                        timestamp=now
                    )

            # 2. Baseline is established, calculate lateral shift ratio
            shift_ratio = float(abs(center_x - self.baseline_center_x))
            triggered = shift_ratio > self.lateral_shift_ratio

            # Confidence is scaled based on how far shift_ratio exceeds the threshold
            if triggered:
                excess = shift_ratio - self.lateral_shift_ratio
                confidence = float(min(0.99, 0.5 + excess / 0.3))
            else:
                confidence = 0.0

            # 3. Track stability for re-baselining
            self.stability_buffer.append((now, center_x))

            # Check if we have history covering at least the required duration
            if now - self.stability_buffer[0][0] >= self.rebaseline_duration_sec:
                # Get only the samples within the duration window
                cutoff = now - self.rebaseline_duration_sec
                recent_samples = [item for item in self.stability_buffer if item[0] >= cutoff]

                if recent_samples:
                    centers = [item[1] for item in recent_samples]
                    center_range = max(centers) - min(centers)
                    if center_range <= self.rebaseline_stability_threshold:
                        # User has stabilized at a new posture for >= 10 seconds, re-baseline
                        new_baseline = float(sum(centers) / len(centers))
                        logger.info(f"Sustained stable posture detected. Re-baselining shoulder pose to: {new_baseline:.4f}")
                        self.baseline_center_x = new_baseline
                        self.stability_buffer.clear()
                        # Update current frame results since we have re-baselined
                        triggered = False
                        confidence = 0.0
                        shift_ratio = 0.0

            # Clean up old samples to avoid memory growth
            if self.stability_buffer:
                cutoff = now - self.rebaseline_duration_sec
                keep_idx = 0
                for i, item in enumerate(self.stability_buffer):
                    if item[0] < cutoff:
                        keep_idx = i
                    else:
                        break
                self.stability_buffer = self.stability_buffer[keep_idx:]

            return DetectionResult(
                signal_name="shoulder_pose",
                triggered=triggered,
                confidence=confidence,
                details={
                    "left_shoulder_visibility": left_vis,
                    "right_shoulder_visibility": right_vis,
                    "left_shoulder_x": float(left_shoulder.x),
                    "right_shoulder_x": float(right_shoulder.x),
                    "shoulder_center_x": center_x,
                    "baseline_center_x": self.baseline_center_x,
                    "lateral_shift_ratio": shift_ratio,
                    "baseline_status": "established"
                },
                timestamp=now
            )

        except Exception as e:
            logger.error(f"Error calculating shoulder pose: {e}")
            return self._create_no_pose_result(now, error_msg=f"Calculation error: {e}")

    def _create_no_pose_result(self, timestamp: float, error_msg: str = "") -> DetectionResult:
        """Helper to create a default result when no pose is detected or on errors."""
        details = {
            "left_shoulder_visibility": 0.0,
            "right_shoulder_visibility": 0.0,
            "baseline_status": "established" if self.baseline_center_x is not None else "collecting"
        }
        if self.baseline_center_x is not None:
            details["baseline_center_x"] = self.baseline_center_x

        if error_msg:
            details["error"] = error_msg
        else:
            details["error"] = "No pose detected"

        return DetectionResult(
            signal_name="shoulder_pose",
            triggered=False,
            confidence=0.0,
            details=details,
            timestamp=timestamp
        )


def cv2_cvtColor_safe(frame: np.ndarray) -> np.ndarray:
    """Safe wrapper around cv2.cvtColor to avoid unresolved import issues."""
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
