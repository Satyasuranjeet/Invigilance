import os
import time
import yaml
import logging
import numpy as np
import cv2
from src.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


class HeadPoseDetector:
    """Estimates head yaw, pitch, and roll using MediaPipe Face Mesh landmarks and cv2.solvePnP.

    Known limitations:
        - Accuracy drops at extreme angles (>60° yaw) where landmarks become occluded or unreliable.
        - Fast motion or low light can lead to motion blur or landmark jitter, impacting pose stability.
        - Intrinsics are estimated using frame size, which is an approximation and might introduce systematic error.
        - Does not account for asymmetric facial features without subject-specific calibration.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml") -> None:
        """Initializes the HeadPoseDetector, loading config thresholds and initializing Face Mesh.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Default fallback values
        self.yaw_threshold_deg: float = 30.0
        self.pitch_threshold_deg: float = 20.0

        # Load config
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config and "head_pose" in config:
                    hp_cfg = config["head_pose"]
                    self.yaw_threshold_deg = float(hp_cfg.get("yaw_threshold_deg", 30.0))
                    self.pitch_threshold_deg = float(hp_cfg.get("pitch_threshold_deg", 20.0))
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

        # Define 3D model points of a generic face in OpenCV coordinate system
        # (X points right, Y points down, Z points inside camera/forward)
        # Nose tip at origin, other points relative to it
        self.face_3d = np.array([
            [0.0, 0.0, 0.0],          # Nose tip (landmark 1)
            [0.0, 330.0, -65.0],      # Chin (landmark 152)
            [225.0, -170.0, -135.0],  # Left eye outer corner (landmark 263)
            [-225.0, -170.0, -135.0], # Right eye outer corner (landmark 33)
            [150.0, 150.0, -125.0],   # Left mouth corner (landmark 291)
            [-150.0, 150.0, -125.0]   # Right mouth corner (landmark 61)
        ], dtype=np.float64)

    def analyze(self, frame: np.ndarray, face_mesh_results = None) -> DetectionResult:
        """Analyzes a video frame to estimate head pose and detect sustained looking away.

        Args:
            frame (np.ndarray): Input video frame in BGR format.
            face_mesh_results (Optional[Any]): Pre-processed Face Mesh results.

        Returns:
            DetectionResult: The head pose detection results.
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
            return self._create_no_face_result(now)

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        try:
            # Extract 2D coordinates for the 6 key facial landmarks
            # Nose Tip: 1, Chin: 152, Left Eye Outer: 263, Right Eye Outer: 33, Left Mouth Corner: 291, Right Mouth Corner: 61
            indices = [1, 152, 263, 33, 291, 61]
            face_2d = []
            for idx in indices:
                lm = landmarks[idx]
                face_2d.append([lm.x * w, lm.y * h])
            face_2d = np.array(face_2d, dtype=np.float64)

            # Approximate camera matrix assuming no lens distortion
            focal_length = w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP to get rotation (rvec) and translation (tvec) vectors
            success, rvec, tvec = cv2.solvePnP(self.face_3d, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            if not success:
                return self._create_no_face_result(now, error_msg="solvePnP failed to converge")

            # Convert rotation vector to 3x3 rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Decompose the rotation matrix to extract Euler angles
            # euler_angles returns: [pitch (X), yaw (Y), roll (Z)] in degrees
            euler_angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            pitch = float(euler_angles[0])
            yaw = float(euler_angles[1])
            roll = float(euler_angles[2])

            # Check thresholds
            triggered = False
            yaw_diff = abs(yaw) - self.yaw_threshold_deg
            pitch_diff = abs(pitch) - self.pitch_threshold_deg

            if yaw_diff > 0 or pitch_diff > 0:
                triggered = True

            # Confidence formula:
            # Scales from 0.5 up to 0.99 based on how far the head pose exceeds the threshold
            # Formula: min(0.99, 0.5 + max(yaw_diff, pitch_diff) / 30.0) if triggered, else 0.0
            if triggered:
                excess = max(yaw_diff, pitch_diff)
                confidence = float(min(0.99, 0.5 + excess / 30.0))
            else:
                confidence = 0.0

            details = {
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "roll_deg": roll,
                "translation_vec": tvec.flatten().tolist()
            }

            return DetectionResult(
                signal_name="head_pose",
                triggered=triggered,
                confidence=confidence,
                details=details,
                timestamp=now
            )

        except Exception as e:
            logger.error(f"Error calculating head pose: {e}")
            return self._create_no_face_result(now, error_msg=f"Pose calculation error: {e}")

    def _create_no_face_result(self, timestamp: float, error_msg: str = "") -> DetectionResult:
        """Helper to create a default result when no face is detected or on errors."""
        details = {
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "translation_vec": [0.0, 0.0, 0.0]
        }
        if error_msg:
            details["error"] = error_msg
        else:
            details["error"] = "No face detected"

        return DetectionResult(
            signal_name="head_pose",
            triggered=False,
            confidence=0.0,
            details=details,
            timestamp=timestamp
        )


def cv2_cvtColor_safe(frame: np.ndarray) -> np.ndarray:
    """Safe wrapper around cv2.cvtColor to avoid unresolved import issues."""
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
