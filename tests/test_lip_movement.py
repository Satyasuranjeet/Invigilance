import os
import time
import urllib.request
import cv2
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.detectors.lip_movement import LipMovementDetector
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


def test_lip_movement_config_loading():
    """Verifies that the detector loads parameters from config or falls back to defaults."""
    # Test with default config path which exists
    detector = LipMovementDetector()
    assert detector.movement_var_threshold == 0.02
    assert detector.buffer_duration_sec == 1.5

    # Test with non-existent config path
    detector_def = LipMovementDetector(config_path="non_existent.yaml")
    assert detector_def.movement_var_threshold == 0.02
    assert detector_def.buffer_duration_sec == 1.5


def test_silent_still_no_trigger(portrait_straight):
    """Verifies that a static portrait (silent/still) does not trigger lip movement."""
    detector = LipMovementDetector()
    
    # We analyze the same static frame multiple times with time progressing
    # and verify that it never triggers because MAR remains constant (variance = 0)
    start_time = 100.0
    for i in range(10):
        t = start_time + (i * 0.1)
        with patch("time.time", return_value=t):
            result = detector.analyze(portrait_straight)
            assert isinstance(result, DetectionResult)
            assert result.signal_name == "lip_movement"
            assert result.triggered is False
            assert result.confidence == 0.0
            assert result.details["mar_variance"] < detector.movement_var_threshold


def test_no_face_detected_handling():
    """Verifies behavior when no face is present in the frame."""
    detector = LipMovementDetector()
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Fill the buffer first to verify it gets cleared
    detector.mar_buffer = [(100.0, 0.1), (101.0, 0.2)]
    
    result = detector.analyze(empty_frame)

    assert result.triggered is False
    assert result.confidence == 0.0
    assert result.details["error"] == "No face detected"
    assert len(detector.mar_buffer) == 0


def test_lip_movement_talking_trigger():
    """Verifies that oscillating MAR (talking) triggers the detector after min_duration_sec."""
    detector = LipMovementDetector()

    # Create mock landmarks that alternate between closed and open mouth
    # Landmarks: 13, 14, 78, 308
    mock_mesh_results_closed = MagicMock()
    mock_landmarks_closed = [MagicMock(x=0.5, y=0.5) for _ in range(478)]
    # MAR = dist(13, 14)/dist(78, 308)
    # Closed: 13 and 14 are close
    mock_landmarks_closed[13].y = 0.50
    mock_landmarks_closed[14].y = 0.50
    mock_landmarks_closed[78].x = 0.45
    mock_landmarks_closed[308].x = 0.55
    mock_mesh_results_closed.multi_face_landmarks = [MagicMock(landmark=mock_landmarks_closed)]

    mock_mesh_results_open = MagicMock()
    mock_landmarks_open = [MagicMock(x=0.5, y=0.5) for _ in range(478)]
    # Open: 13 and 14 are apart
    mock_landmarks_open[13].y = 0.46
    mock_landmarks_open[14].y = 0.54
    mock_landmarks_open[78].x = 0.45
    mock_landmarks_open[308].x = 0.55
    mock_mesh_results_open.multi_face_landmarks = [MagicMock(landmark=mock_landmarks_open)]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    start_time = 100.0

    # Feed alternating frames
    for i in range(10):
        t = start_time + (i * 0.1)
        mock_res = mock_mesh_results_open if i % 2 == 0 else mock_mesh_results_closed
        detector.face_mesh.process = MagicMock(return_value=mock_res)
        
        with patch("time.time", return_value=t):
            result = detector.analyze(frame)
            
            # Should only trigger after min_duration_sec (0.5s, i.e., index >= 5)
            if i < 5:
                assert result.triggered is False
            else:
                assert result.triggered is True
                assert result.confidence >= 0.5
                assert result.details["mar_variance"] > detector.movement_var_threshold


def test_lip_movement_yawning_no_trigger():
    """Verifies that sustained open mouth (yawn) does not trigger once the buffer stabilizes."""
    detector = LipMovementDetector()

    # Create mock landmarks for yawning (held wide open mouth)
    mock_mesh_results_open = MagicMock()
    mock_landmarks_open = [MagicMock(x=0.5, y=0.5) for _ in range(478)]
    mock_landmarks_open[13].y = 0.46
    mock_landmarks_open[14].y = 0.54
    mock_landmarks_open[78].x = 0.45
    mock_landmarks_open[308].x = 0.55
    mock_mesh_results_open.multi_face_landmarks = [MagicMock(landmark=mock_landmarks_open)]
    detector.face_mesh.process = MagicMock(return_value=mock_mesh_results_open)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    start_time = 100.0

    # Step 1: Pre-populate buffer with initial closed state to simulate starting silent,
    # then yawning.
    mock_mesh_results_closed = MagicMock()
    mock_landmarks_closed = [MagicMock(x=0.5, y=0.5) for _ in range(478)]
    mock_landmarks_closed[13].y = 0.50
    mock_landmarks_closed[14].y = 0.50
    mock_landmarks_closed[78].x = 0.45
    mock_landmarks_closed[308].x = 0.55
    mock_mesh_results_closed.multi_face_landmarks = [MagicMock(landmark=mock_landmarks_closed)]

    # Closed mouth for 1.0 second
    for i in range(10):
        t = start_time + (i * 0.1)
        detector.face_mesh.process = MagicMock(return_value=mock_mesh_results_closed)
        with patch("time.time", return_value=t):
            result = detector.analyze(frame)
            assert result.triggered is False

    # Step 2: Open mouth and hold it open (yawn) for 2.0 seconds
    # (MAR goes from 0.0 to 0.8, creating a transient variance, but because there is no
    # oscillation, it should never trigger)
    yawn_start = start_time + 1.0
    for i in range(20):
        t = yawn_start + (i * 0.1)
        detector.face_mesh.process = MagicMock(return_value=mock_mesh_results_open)
        with patch("time.time", return_value=t):
            result = detector.analyze(frame)
            
            # It should never trigger during or after the yawn transition
            assert result.triggered is False
            
            # Once the buffer is filled only with open mouth frames (takes 1.5 seconds),
            # variance must drop to 0.
            if t - yawn_start > 1.5:
                assert result.details["mar_variance"] < detector.movement_var_threshold
