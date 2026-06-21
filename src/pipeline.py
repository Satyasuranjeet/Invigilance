import logging
import time
from typing import List
import numpy as np

from src.detectors.base import DetectionResult
from src.detectors.eye_gaze import EyeGazeDetector
from src.detectors.head_pose import HeadPoseDetector
from src.detectors.shoulder_pose import ShoulderPoseDetector
from src.detectors.lip_movement import LipMovementDetector
from src.detectors.face_presence import FacePresenceDetector
from src.detectors.object_detector import ObjectDetector

logger = logging.getLogger(__name__)


class ProctorPipeline:
    """Orchestrates all proctoring detectors on a per-frame basis.

    Known Limitations:
        - Detectors run sequentially on a single thread.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml", model_path: str = "models/yolov8n.pt") -> None:
        """Initializes all detector modules.

        Args:
            config_path (str): Path to thresholds.yaml.
            model_path (str): Path to the YOLOv8n model weights.
        """
        self.config_path = config_path
        self.model_path = model_path

        # Initialize shared MediaPipe Face Mesh
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.shared_face_mesh = self.mp_face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=3,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            logger.error(f"Failed to initialize shared MediaPipe Face Mesh: {e}")
            raise

        # Initialize each detector. In case of any startup error, let it raise so main knows.
        self.eye_gaze_detector = EyeGazeDetector(config_path=self.config_path)
        self.head_pose_detector = HeadPoseDetector(config_path=self.config_path)
        self.shoulder_pose_detector = ShoulderPoseDetector(config_path=self.config_path)
        self.lip_movement_detector = LipMovementDetector(config_path=self.config_path)
        self.face_presence_detector = FacePresenceDetector(config_path=self.config_path)
        self.object_detector = ObjectDetector(config_path=self.config_path, model_path=self.model_path)

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Processes a video frame through all detectors in a fixed order.

        Each detector call is wrapped in a try/except block to ensure that if a single
        detector fails or throws an exception, the rest of the pipeline continues unimpeded.

        Args:
            frame (np.ndarray): The video frame to process.

        Returns:
            List[DetectionResult]: Collected detection results from all detectors.
        """
        results: List[DetectionResult] = []
        now = time.time()

        # Run Face Mesh once per frame and share results
        face_mesh_results = None
        try:
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_mesh_results = self.shared_face_mesh.process(rgb_frame)
        except Exception as e:
            logger.error(f"Shared Face Mesh processing failed: {e}", exc_info=True)

        # 1. Eye Gaze Tracking
        try:
            results.append(self.eye_gaze_detector.analyze(frame, face_mesh_results=face_mesh_results))
        except Exception as e:
            logger.error(f"EyeGazeDetector failed during processing: {e}", exc_info=True)
            results.append(
                DetectionResult(
                    signal_name="eye_gaze",
                    triggered=False,
                    confidence=0.0,
                    details={"error": f"Pipeline failure: {type(e).__name__}: {e}"},
                    timestamp=now,
                )
            )

        # 2. Head Pose Estimation
        try:
            results.append(self.head_pose_detector.analyze(frame, face_mesh_results=face_mesh_results))
        except Exception as e:
            logger.error(f"HeadPoseDetector failed during processing: {e}", exc_info=True)
            results.append(
                DetectionResult(
                    signal_name="head_pose",
                    triggered=False,
                    confidence=0.0,
                    details={"error": f"Pipeline failure: {type(e).__name__}: {e}"},
                    timestamp=now,
                )
            )

        # 3. Shoulder Movement Tracking
        try:
            results.append(self.shoulder_pose_detector.analyze(frame))
        except Exception as e:
            logger.error(f"ShoulderPoseDetector failed during processing: {e}", exc_info=True)
            results.append(
                DetectionResult(
                    signal_name="shoulder_pose",
                    triggered=False,
                    confidence=0.0,
                    details={"error": f"Pipeline failure: {type(e).__name__}: {e}"},
                    timestamp=now,
                )
            )

        # 4. Lip Movement Detection
        try:
            results.append(self.lip_movement_detector.analyze(frame, face_mesh_results=face_mesh_results))
        except Exception as e:
            logger.error(f"LipMovementDetector failed during processing: {e}", exc_info=True)
            results.append(
                DetectionResult(
                    signal_name="lip_movement",
                    triggered=False,
                    confidence=0.0,
                    details={"error": f"Pipeline failure: {type(e).__name__}: {e}"},
                    timestamp=now,
                )
            )

        # 5. Face Presence / Multi-face Check
        try:
            results.append(self.face_presence_detector.analyze(frame, face_mesh_results=face_mesh_results))
        except Exception as e:
            logger.error(f"FacePresenceDetector failed during processing: {e}", exc_info=True)
            results.append(
                DetectionResult(
                    signal_name="face_presence",
                    triggered=False,
                    confidence=0.0,
                    details={"error": f"Pipeline failure: {type(e).__name__}: {e}"},
                    timestamp=now,
                )
            )

        # 6. Object Detection (YOLOv8n) - returns a list of DetectionResults
        try:
            results.extend(self.object_detector.analyze(frame))
        except Exception as e:
            logger.error(f"ObjectDetector failed during processing: {e}", exc_info=True)
            # Add safe fallbacks for each of the three signals emitted by ObjectDetector
            results.extend(
                [
                    DetectionResult(
                        signal_name="person_count",
                        triggered=True,  # Safety fallback
                        confidence=0.0,
                        details={"count": 0, "boxes": [], "error": f"Pipeline failure: {type(e).__name__}: {e}"},
                        timestamp=now,
                    ),
                    DetectionResult(
                        signal_name="phone_detected",
                        triggered=False,
                        confidence=0.0,
                        details={"count": 0, "boxes": [], "error": f"Pipeline failure: {type(e).__name__}: {e}"},
                        timestamp=now,
                    ),
                    DetectionResult(
                        signal_name="secondary_device_detected",
                        triggered=False,
                        confidence=0.0,
                        details={
                            "phone_count": 0,
                            "laptop_count": 0,
                            "boxes": [],
                            "error": f"Pipeline failure: {type(e).__name__}: {e}",
                        },
                        timestamp=now,
                    ),
                ]
            )

        return results
