import os
import urllib.request
import cv2
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.detectors.shoulder_pose import ShoulderPoseDetector
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


def create_mock_shoulder_landmarks(left_x: float, left_vis: float, right_x: float, right_vis: float):
    """Helper to create mock pose landmarks list of length 33."""
    mock_left = MagicMock(x=left_x, visibility=left_vis)
    mock_right = MagicMock(x=right_x, visibility=right_vis)
    landmarks = [MagicMock() for _ in range(33)]
    landmarks[11] = mock_left  # LEFT_SHOULDER
    landmarks[12] = mock_right # RIGHT_SHOULDER
    return landmarks


def test_shoulder_pose_config_loading():
    """Verifies that the detector loads parameters from config or falls back to defaults."""
    # Test with default config path which exists
    detector = ShoulderPoseDetector()
    assert detector.lateral_shift_ratio == 0.15
    assert detector.min_visibility == 0.5

    # Test with non-existent config path
    detector_def = ShoulderPoseDetector(config_path="non_existent.yaml")
    assert detector_def.lateral_shift_ratio == 0.15
    assert detector_def.min_visibility == 0.5


def test_shoulder_baseline_collection():
    """Verifies baseline collection frames and initial behavior."""
    detector = ShoulderPoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Center is at 0.5 (left x = 0.4, right x = 0.6)
    mock_landmarks = create_mock_shoulder_landmarks(0.4, 0.9, 0.6, 0.9)
    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = mock_landmarks
    detector.pose.process = MagicMock(return_value=mock_results)

    for i in range(29):
        result = detector.analyze(frame)
        assert result.triggered is False
        assert detector.baseline_center_x is None
        assert result.details["baseline_status"] == "collecting"

    # 30th frame completes the baseline
    result = detector.analyze(frame)
    assert detector.baseline_center_x == 0.5
    assert result.details["baseline_status"] == "established"
    assert result.triggered is False


def test_shoulder_shift_trigger():
    """Verifies that shifting the shoulder position past the threshold triggers the alert."""
    detector = ShoulderPoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Establish baseline at 0.5
    mock_landmarks_center = create_mock_shoulder_landmarks(0.4, 0.9, 0.6, 0.9)
    mock_results_center = MagicMock()
    mock_results_center.pose_landmarks.landmark = mock_landmarks_center
    detector.pose.process = MagicMock(return_value=mock_results_center)
    for _ in range(30):
        detector.analyze(frame)

    assert detector.baseline_center_x == 0.5

    # Center shifts to 0.3 (left x = 0.2, right x = 0.4)
    # Shift = abs(0.3 - 0.5) = 0.2 > 0.15 (lateral_shift_ratio threshold)
    mock_landmarks_shifted = create_mock_shoulder_landmarks(0.2, 0.9, 0.4, 0.9)
    mock_results_shifted = MagicMock()
    mock_results_shifted.pose_landmarks.landmark = mock_landmarks_shifted
    detector.pose.process = MagicMock(return_value=mock_results_shifted)

    result = detector.analyze(frame)
    assert result.triggered is True
    assert result.confidence >= 0.5
    assert result.details["lateral_shift_ratio"] == pytest.approx(0.2)

    # Center shifts to 0.45
    # Shift = abs(0.45 - 0.5) = 0.05 <= 0.15
    mock_landmarks_minor_shift = create_mock_shoulder_landmarks(0.35, 0.9, 0.55, 0.9)
    mock_results_minor_shift = MagicMock()
    mock_results_minor_shift.pose_landmarks.landmark = mock_landmarks_minor_shift
    detector.pose.process = MagicMock(return_value=mock_results_minor_shift)

    result = detector.analyze(frame)
    assert result.triggered is False
    assert result.confidence == 0.0


def test_shoulder_low_visibility():
    """Verifies that low landmark visibility triggers the alert."""
    detector = ShoulderPoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Left shoulder visibility = 0.2, right = 0.9
    mock_landmarks = create_mock_shoulder_landmarks(0.4, 0.2, 0.6, 0.9)
    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = mock_landmarks
    detector.pose.process = MagicMock(return_value=mock_results)

    result = detector.analyze(frame)
    assert result.triggered is True
    assert "Low shoulder visibility" in result.details["error"]
    assert result.confidence >= 0.5


def test_no_pose_detected_handling():
    """Verifies behavior when no pose landmarks are detected at all."""
    detector = ShoulderPoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_results = MagicMock(pose_landmarks=None)
    detector.pose.process = MagicMock(return_value=mock_results)

    result = detector.analyze(frame)
    assert result.triggered is False
    assert result.confidence == 0.0
    assert result.details["error"] == "No pose detected"


def test_shoulder_rebaselining():
    """Verifies that holding a shifted posture stably for >= 10s re-baselines the detector."""
    detector = ShoulderPoseDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Establish baseline at 0.5 at t = 100.0
    mock_landmarks_center = create_mock_shoulder_landmarks(0.4, 0.9, 0.6, 0.9)
    mock_results_center = MagicMock()
    mock_results_center.pose_landmarks.landmark = mock_landmarks_center
    detector.pose.process = MagicMock(return_value=mock_results_center)

    start_time = 100.0
    with patch("time.time", return_value=start_time):
        for _ in range(30):
            detector.analyze(frame)
    assert detector.baseline_center_x == 0.5

    # Shift to 0.3 (triggers alert)
    mock_landmarks_shifted = create_mock_shoulder_landmarks(0.2, 0.9, 0.4, 0.9)
    mock_results_shifted = MagicMock()
    mock_results_shifted.pose_landmarks.landmark = mock_landmarks_shifted
    detector.pose.process = MagicMock(return_value=mock_results_shifted)

    # Frame at t = 100.0 (shifted)
    with patch("time.time", return_value=start_time):
        result = detector.analyze(frame)
        assert result.triggered is True

    # Frame at t = 105.0 (still shifted, stable at 0.3)
    with patch("time.time", return_value=start_time + 5.0):
        result = detector.analyze(frame)
        assert result.triggered is True
        assert detector.baseline_center_x == 0.5

    # Frame at t = 111.0 (held stable at 0.3 for 11 seconds)
    with patch("time.time", return_value=start_time + 11.0):
        result = detector.analyze(frame)
        # Should now be re-baselined to 0.3, and triggered should reset to False
        assert detector.baseline_center_x == pytest.approx(0.3)
        assert result.triggered is False
        assert result.confidence == 0.0


def test_real_image_straight(portrait_straight):
    """Verifies detector execution on a real portrait image fixture."""
    detector = ShoulderPoseDetector()
    # Establish baseline using the real image frame
    for _ in range(30):
        result = detector.analyze(portrait_straight)

    assert detector.baseline_center_x is not None
    assert result.triggered is False
    assert result.details.get("error") is None
