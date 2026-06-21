import os
import time
import urllib.request
import cv2
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.detectors.eye_gaze import EyeGazeDetector
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


@pytest.fixture
def portrait_away():
    url = "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=640"
    filepath = download_fixture_if_missing("portrait_away.jpg", url)
    frame = cv2.imread(filepath)
    assert frame is not None, "Failed to load portrait_away frame"
    return frame


def test_eye_gaze_config_loading():
    """Verifies that the detector loads parameters from config or falls back to defaults."""
    # Test with default config path which exists
    detector = EyeGazeDetector()
    assert detector.away_angle_deg == 25.0
    assert detector.away_duration_sec == 2.0

    # Test with non-existent config path
    detector_def = EyeGazeDetector(config_path="non_existent.yaml")
    assert detector_def.away_angle_deg == 25.0
    assert detector_def.away_duration_sec == 2.0


def test_straight_gaze_no_trigger(portrait_straight):
    """Verifies that a person looking straight at the screen does not trigger the look-away alert."""
    detector = EyeGazeDetector()
    result = detector.analyze(portrait_straight)

    assert isinstance(result, DetectionResult)
    assert result.signal_name == "eye_gaze"
    assert result.triggered is False
    assert result.confidence == 0.0
    assert result.details.get("error") is None or result.details.get("error") == ""
    assert result.details["gaze_angle_deg"] < detector.away_angle_deg


def test_no_face_detected_handling():
    """Verifies behavior when no face is present in the frame."""
    detector = EyeGazeDetector()
    # Create an empty black image (no face)
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.analyze(empty_frame)

    assert result.triggered is False
    assert result.confidence == 0.0
    assert "No face detected" in result.details["error"]


def test_look_away_rolling_timer_trigger():
    """Verifies that the look-away alert triggers only after exceeding away_duration_sec continuously."""
    detector = EyeGazeDetector()

    # We will mock the Face Mesh output to return a landmark list corresponding to "looking far left".
    # We create mock landmark objects.
    mock_landmarks = [MagicMock() for _ in range(478)]
    
    # Set coordinates for right eye: outer 33, inner 133, top 159, bottom 145, iris 468
    # If looking far left of screen (larger x), iris.x is close to or past inner.x (for right eye)
    mock_landmarks[33].x = 0.30
    mock_landmarks[133].x = 0.40
    mock_landmarks[468].x = 0.50  # Far left (exceeds inner corner)
    
    mock_landmarks[159].y = 0.33
    mock_landmarks[145].y = 0.35
    mock_landmarks[468].y = 0.34
    
    # Left eye: outer 263, inner 362, top 386, bottom 374, iris 473
    # If looking far left of screen (larger x), iris.x is close to or past outer.x
    mock_landmarks[362].x = 0.55
    mock_landmarks[263].x = 0.65
    mock_landmarks[473].x = 0.75  # Far left (exceeds outer corner)
    
    mock_landmarks[386].y = 0.33
    mock_landmarks[374].y = 0.35
    mock_landmarks[473].y = 0.34

    # Setup the mock face mesh results
    mock_mesh_results = MagicMock()
    mock_mesh_results.multi_face_landmarks = [MagicMock(landmark=mock_landmarks)]
    detector.face_mesh.process = MagicMock(return_value=mock_mesh_results)

    # Mock time to control the rolling timer precisely
    start_time = 100.0
    
    # Frame 1 at t = 100.0 (first look away)
    with patch("time.time", return_value=start_time):
        result1 = detector.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result1.details["gaze_angle_deg"] > detector.away_angle_deg
        # Should not trigger immediately on first frame
        assert result1.triggered is False
        assert result1.confidence == 0.0

    # Frame 2 at t = 101.0 (1.0s look away)
    with patch("time.time", return_value=start_time + 1.0):
        result2 = detector.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result2.triggered is False

    # Frame 3 at t = 102.5 (2.5s look away - exceeds the 2.0s duration)
    with patch("time.time", return_value=start_time + 2.5):
        result3 = detector.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result3.triggered is True
        assert result3.confidence > 0.5  # Should derive confidence score > 0.5

    # Frame 4 at t = 103.0 with user looking back at the screen (gaze angle < threshold)
    # Set iris back to center (0.35 and 0.60)
    mock_landmarks[468].x = 0.35
    mock_landmarks[473].x = 0.60
    with patch("time.time", return_value=start_time + 3.0):
        result4 = detector.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result4.triggered is False
        assert result4.confidence == 0.0
        assert detector.first_look_away_time is None
