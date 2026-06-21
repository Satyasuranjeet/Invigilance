import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.detectors.base import DetectionResult
from src.pipeline import ProctorPipeline


class TestProctorPipeline(unittest.TestCase):
    """Unit tests for the ProctorPipeline class."""

    @patch("src.pipeline.EyeGazeDetector")
    @patch("src.pipeline.HeadPoseDetector")
    @patch("src.pipeline.ShoulderPoseDetector")
    @patch("src.pipeline.LipMovementDetector")
    @patch("src.pipeline.FacePresenceDetector")
    @patch("src.pipeline.ObjectDetector")
    def test_pipeline_initialization(
        self,
        mock_object,
        mock_face,
        mock_lip,
        mock_shoulder,
        mock_head,
        mock_eye,
    ):
        """Verifies that the pipeline initializes all expected detectors."""
        pipeline = ProctorPipeline(config_path="config/thresholds.yaml", model_path="models/yolov8n.pt")

        mock_eye.assert_called_once_with(config_path="config/thresholds.yaml")
        mock_head.assert_called_once_with(config_path="config/thresholds.yaml")
        mock_shoulder.assert_called_once_with(config_path="config/thresholds.yaml")
        mock_lip.assert_called_once_with(config_path="config/thresholds.yaml")
        mock_face.assert_called_once_with(config_path="config/thresholds.yaml")
        mock_object.assert_called_once_with(
            config_path="config/thresholds.yaml", model_path="models/yolov8n.pt"
        )

        self.assertIsNotNone(pipeline.eye_gaze_detector)
        self.assertIsNotNone(pipeline.head_pose_detector)
        self.assertIsNotNone(pipeline.shoulder_pose_detector)
        self.assertIsNotNone(pipeline.lip_movement_detector)
        self.assertIsNotNone(pipeline.face_presence_detector)
        self.assertIsNotNone(pipeline.object_detector)

    @patch("src.pipeline.EyeGazeDetector")
    @patch("src.pipeline.HeadPoseDetector")
    @patch("src.pipeline.ShoulderPoseDetector")
    @patch("src.pipeline.LipMovementDetector")
    @patch("src.pipeline.FacePresenceDetector")
    @patch("src.pipeline.ObjectDetector")
    def test_process_frame_success(
        self,
        mock_object,
        mock_face,
        mock_lip,
        mock_shoulder,
        mock_head,
        mock_eye,
    ):
        """Verifies that process_frame successfully aggregates results from all detectors."""
        # Setup mocks to return dummy DetectionResult objects
        mock_eye.return_value.analyze.return_value = DetectionResult(
            signal_name="eye_gaze", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_head.return_value.analyze.return_value = DetectionResult(
            signal_name="head_pose", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_shoulder.return_value.analyze.return_value = DetectionResult(
            signal_name="shoulder_pose", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_lip.return_value.analyze.return_value = DetectionResult(
            signal_name="lip_movement", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_face.return_value.analyze.return_value = DetectionResult(
            signal_name="face_presence", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_object.return_value.analyze.return_value = [
            DetectionResult(
                signal_name="person_count", triggered=False, confidence=0.0, details={}, timestamp=100.0
            ),
            DetectionResult(
                signal_name="phone_detected", triggered=False, confidence=0.0, details={}, timestamp=100.0
            ),
            DetectionResult(
                signal_name="secondary_device_detected",
                triggered=False,
                confidence=0.0,
                details={},
                timestamp=100.0,
            ),
        ]

        pipeline = ProctorPipeline()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = pipeline.process_frame(dummy_frame)

        self.assertEqual(len(results), 8)
        signals = [res.signal_name for res in results]
        expected_signals = [
            "eye_gaze",
            "head_pose",
            "shoulder_pose",
            "lip_movement",
            "face_presence",
            "person_count",
            "phone_detected",
            "secondary_device_detected",
        ]
        self.assertEqual(signals, expected_signals)

    @patch("src.pipeline.EyeGazeDetector")
    @patch("src.pipeline.HeadPoseDetector")
    @patch("src.pipeline.ShoulderPoseDetector")
    @patch("src.pipeline.LipMovementDetector")
    @patch("src.pipeline.FacePresenceDetector")
    @patch("src.pipeline.ObjectDetector")
    def test_process_frame_resilience_to_detector_exception(
        self,
        mock_object,
        mock_face,
        mock_lip,
        mock_shoulder,
        mock_head,
        mock_eye,
    ):
        """Verifies that if one detector raises an exception, the pipeline doesn't crash."""
        # EyeGazeDetector will fail
        mock_eye.return_value.analyze.side_effect = RuntimeError("Camera calibration missing")

        # Others succeed
        mock_head.return_value.analyze.return_value = DetectionResult(
            signal_name="head_pose", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_shoulder.return_value.analyze.return_value = DetectionResult(
            signal_name="shoulder_pose", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_lip.return_value.analyze.return_value = DetectionResult(
            signal_name="lip_movement", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_face.return_value.analyze.return_value = DetectionResult(
            signal_name="face_presence", triggered=False, confidence=0.0, details={}, timestamp=100.0
        )
        mock_object.return_value.analyze.return_value = [
            DetectionResult(
                signal_name="person_count", triggered=False, confidence=0.0, details={}, timestamp=100.0
            ),
            DetectionResult(
                signal_name="phone_detected", triggered=False, confidence=0.0, details={}, timestamp=100.0
            ),
            DetectionResult(
                signal_name="secondary_device_detected",
                triggered=False,
                confidence=0.0,
                details={},
                timestamp=100.0,
            ),
        ]

        pipeline = ProctorPipeline()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = pipeline.process_frame(dummy_frame)

        self.assertEqual(len(results), 8)

        # First result should be the fallback for eye_gaze
        eye_gaze_res = results[0]
        self.assertEqual(eye_gaze_res.signal_name, "eye_gaze")
        self.assertFalse(eye_gaze_res.triggered)
        self.assertEqual(eye_gaze_res.confidence, 0.0)
        self.assertIn("error", eye_gaze_res.details)
        self.assertIn("RuntimeError", eye_gaze_res.details["error"])

        # Second result (head_pose) should have executed normally
        head_pose_res = results[1]
        self.assertEqual(head_pose_res.signal_name, "head_pose")
        self.assertFalse(head_pose_res.triggered)
