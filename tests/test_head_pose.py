import os
import urllib.request
import cv2
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.detectors.head_pose import HeadPoseDetector
from src.detectors.base import DetectionResult

FIXTURES_DIR = os.path.join("tests", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)


def download_fixture_if_missing(filename: str, url: str) -> str:
    """Helper to download a test fixture if it does not exist."""
    filepath = os.path.join(FIXTURES_DIR, filename)
    if not os.path.exists(filepath):
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
    return filepath


@pytest.fixture
def portrait_straight():
    url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640"
    filepath = download_fixture_if_missing("portrait_straight.jpg", url)
    frame = cv2.imread(filepath)
    assert frame is not None, "Failed to load portrait_straight frame"
    return frame


def test_head_pose_config_loading():
    """Verifies that the detector loads parameters from config or falls back to defaults."""
    # Test with default config path which exists
    detector = HeadPoseDetector()
    assert detector.yaw_threshold_deg == 30.0
    assert detector.pitch_threshold_deg == 20.0

    # Test with non-existent config path
    detector_def = HeadPoseDetector(config_path="non_existent.yaml")
    assert detector_def.yaw_threshold_deg == 30.0
    assert detector_def.pitch_threshold_deg == 20.0


def test_straight_head_pose_no_trigger(portrait_straight):
    """Verifies that a person looking straight at the screen does not trigger head pose alert."""
    detector = HeadPoseDetector()
    result = detector.analyze(portrait_straight)

    assert isinstance(result, DetectionResult)
    assert result.signal_name == "head_pose"
    assert result.triggered is False
    assert result.confidence == 0.0
    assert result.details.get("error") is None
    assert abs(result.details["yaw_deg"]) < detector.yaw_threshold_deg
    assert abs(result.details["pitch_deg"]) < detector.pitch_threshold_deg


def test_no_face_detected_handling():
    """Verifies behavior when no face is present in the frame."""
    detector = HeadPoseDetector()
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.analyze(empty_frame)

    assert result.triggered is False
    assert result.confidence == 0.0
    assert result.details["error"] == "No face detected"


def test_head_pose_mock_yaw_trigger():
    """Verifies that head pose triggers when yaw exceeds threshold."""
    detector = HeadPoseDetector()

    # We mock Face Mesh landmarks using mathematically derived projection coordinates for yaw = 35 deg.
    mock_landmarks = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(468)]
    
    # Derivation for 35 deg yaw:
    mock_landmarks[1].x = 0.5000
    mock_landmarks[1].y = 0.5000
    
    mock_landmarks[152].x = 0.4606
    mock_landmarks[152].y = 0.9647
    
    mock_landmarks[263].x = 0.6406
    mock_landmarks[263].y = 0.2019
    
    mock_landmarks[33].x = 0.2430
    mock_landmarks[33].y = 0.2774
    
    mock_landmarks[291].x = 0.5631
    mock_landmarks[291].y = 0.7464
    
    mock_landmarks[61].x = 0.3022
    mock_landmarks[61].y = 0.7033

    # Setup the mock face mesh results
    mock_mesh_results = MagicMock()
    mock_mesh_results.multi_face_landmarks = [MagicMock(landmark=mock_landmarks)]
    detector.face_mesh.process = MagicMock(return_value=mock_mesh_results)

    # Call analyze on a 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.analyze(frame)

    # Should trigger because the yaw has a significant shift
    assert result.triggered is True
    assert result.confidence >= 0.5
    assert abs(result.details["yaw_deg"]) > detector.yaw_threshold_deg


def test_head_pose_mock_pitch_trigger():
    """Verifies that head pose triggers when pitch exceeds threshold."""
    detector = HeadPoseDetector()

    # We mock Face Mesh landmarks using mathematically derived projection coordinates for pitch = 35 deg.
    mock_landmarks = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(468)]
    
    # Derivation for 35 deg pitch:
    mock_landmarks[1].x = 0.5000
    mock_landmarks[1].y = 0.5000
    
    mock_landmarks[152].x = 0.5000
    mock_landmarks[152].y = 0.8610
    
    mock_landmarks[263].x = 0.7841
    mock_landmarks[263].y = 0.3959
    
    mock_landmarks[33].x = 0.2159
    mock_landmarks[33].y = 0.3959
    
    mock_landmarks[291].x = 0.6525
    mock_landmarks[291].y = 0.7637
    
    mock_landmarks[61].x = 0.3475
    mock_landmarks[61].y = 0.7637

    mock_mesh_results = MagicMock()
    mock_mesh_results.multi_face_landmarks = [MagicMock(landmark=mock_landmarks)]
    detector.face_mesh.process = MagicMock(return_value=mock_mesh_results)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.analyze(frame)

    assert result.triggered is True
    assert result.confidence >= 0.5
    assert abs(result.details["pitch_deg"]) > detector.pitch_threshold_deg
