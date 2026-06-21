import os
import time
import urllib.request
import cv2
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.detectors.face_presence import FacePresenceDetector
from src.detectors.base import DetectionResult

FIXTURES_DIR = os.path.join("tests", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

IMAGE_URLS = {
    "empty_desk.jpg": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=640",
    "portrait_straight.jpg": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640",
    "two_people.jpg": "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=640"
}


def download_fixture_if_missing(filename: str) -> str:
    """Helper to download a test fixture if it does not exist."""
    filepath = os.path.join(FIXTURES_DIR, filename)
    if not os.path.exists(filepath):
        url = IMAGE_URLS[filename]
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
    return filepath


@pytest.fixture
def empty_desk_frame():
    filepath = download_fixture_if_missing("empty_desk.jpg")
    frame = cv2.imread(filepath)
    assert frame is not None, "Failed to load empty desk frame"
    return frame


@pytest.fixture
def portrait_straight_frame():
    filepath = download_fixture_if_missing("portrait_straight.jpg")
    frame = cv2.imread(filepath)
    assert frame is not None, "Failed to load portrait_straight frame"
    return frame


def test_face_presence_config_loading():
    """Verifies that the detector loads parameters from config or falls back to defaults."""
    # Test default
    detector = FacePresenceDetector()
    assert detector.absence_duration_sec == 1.5

    # Test fallback
    detector_def = FacePresenceDetector(config_path="non_existent.yaml")
    assert detector_def.absence_duration_sec == 1.5


def test_single_face_no_trigger(portrait_straight_frame):
    """Verifies that a frame with exactly 1 face does not trigger face presence alerts."""
    detector = FacePresenceDetector()
    result = detector.analyze(portrait_straight_frame)

    assert isinstance(result, DetectionResult)
    assert result.signal_name == "face_presence"
    assert result.triggered is False
    assert result.confidence >= 0.5
    assert result.details["face_count"] == 1
    assert len(result.details["boxes"]) == 1


def test_multi_face_immediate_trigger():
    """Verifies that a frame with multiple faces (>1) triggers face presence alerts immediately."""
    detector = FacePresenceDetector()
    
    # Mock two faces
    mock_mesh_results = MagicMock()
    mock_face1 = MagicMock(landmark=[MagicMock(x=0.1, y=0.1), MagicMock(x=0.2, y=0.2)])
    mock_face2 = MagicMock(landmark=[MagicMock(x=0.4, y=0.4), MagicMock(x=0.5, y=0.5)])
    mock_mesh_results.multi_face_landmarks = [mock_face1, mock_face2]
    detector.face_mesh.process = MagicMock(return_value=mock_mesh_results)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.analyze(frame)

    assert isinstance(result, DetectionResult)
    assert result.signal_name == "face_presence"
    assert result.triggered is True
    assert result.details["face_count"] == 2
    assert len(result.details["boxes"]) == 2


def test_zero_face_duration_trigger(empty_desk_frame):
    """Verifies that 0 faces requires sustained absence before triggering."""
    detector = FacePresenceDetector()
    start_time = 100.0

    # 1. First frame at t=100.0: should NOT trigger immediately
    with patch("time.time", return_value=start_time):
        result1 = detector.analyze(empty_desk_frame)
        assert result1.triggered is False
        assert result1.details["face_count"] == 0
        assert result1.details["absence_duration"] == 0.0

    # 2. Frame at t=101.0 (1.0s elapsed): should NOT trigger (threshold is 1.5s)
    with patch("time.time", return_value=start_time + 1.0):
        result2 = detector.analyze(empty_desk_frame)
        assert result2.triggered is False
        assert result2.details["absence_duration"] == 1.0

    # 3. Frame at t=101.6 (1.6s elapsed): should trigger
    with patch("time.time", return_value=start_time + 1.6):
        result3 = detector.analyze(empty_desk_frame)
        assert result3.triggered is True
        assert result3.details["absence_duration"] == pytest.approx(1.6)
        assert result3.confidence == 0.0

    # 4. If we detect 1 face again, timer resets and triggered becomes False
    # Mock face mesh output for 1 face
    mock_mesh_results = MagicMock()
    mock_landmark = MagicMock(x=0.5, y=0.5)
    mock_face = MagicMock(landmark=[mock_landmark])
    mock_mesh_results.multi_face_landmarks = [mock_face]
    detector.face_mesh.process = MagicMock(return_value=mock_mesh_results)

    with patch("time.time", return_value=start_time + 2.0):
        # We need a dummy non-empty frame for dimensions shape
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result4 = detector.analyze(dummy_frame)
        assert result4.triggered is False
        assert result4.details["face_count"] == 1
        assert detector.first_absence_time is None


def test_lowest_confidence_multi_face():
    """Verifies that for multi-face cases, confidence is the minimum of face confidences."""
    detector = FacePresenceDetector()
    
    # Mock three faces with specific confidences/scores
    mock_mesh_results = MagicMock()
    
    mock_face1 = MagicMock(landmark=[MagicMock(x=0.1, y=0.1), MagicMock(x=0.2, y=0.2)])
    mock_face1.confidence = 0.85
    
    mock_face2 = MagicMock(landmark=[MagicMock(x=0.4, y=0.4), MagicMock(x=0.5, y=0.5)])
    mock_face2.score = 0.70
    
    mock_face3 = MagicMock(landmark=[MagicMock(x=0.7, y=0.7), MagicMock(x=0.8, y=0.8)])
    mock_face3.confidence = 0.95
    
    mock_mesh_results.multi_face_landmarks = [mock_face1, mock_face2, mock_face3]
    detector.face_mesh.process = MagicMock(return_value=mock_mesh_results)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.analyze(frame)
    
    assert result.triggered is True
    assert result.details["face_count"] == 3
    # Confidence should be the lowest (0.70)
    assert result.confidence == pytest.approx(0.70)
