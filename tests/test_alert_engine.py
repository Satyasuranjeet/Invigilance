import os
import json
import shutil
import tempfile
from src.detectors.base import DetectionResult
from src.event_logger import EventLogger
from src.alert_engine import AlertEngine


def test_severity_tagging():
    """Tests that custom and default severity mappings are parsed correctly."""
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "thresholds.yaml")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("""
severity:
  default:
    medium: 0.3
    high: 0.6
  custom_sig:
    medium: 0.5
    high: 0.9
""")
        engine = AlertEngine(config_path=config_path)

        # Test default mappings
        assert engine.get_severity("other_sig", 0.2) == "low"
        assert engine.get_severity("other_sig", 0.4) == "medium"
        assert engine.get_severity("other_sig", 0.7) == "high"

        # Test custom mappings
        assert engine.get_severity("custom_sig", 0.4) == "low"
        assert engine.get_severity("custom_sig", 0.6) == "medium"
        assert engine.get_severity("custom_sig", 0.95) == "high"
    finally:
        shutil.rmtree(temp_dir)


def test_alert_formatting():
    """Tests formatting of human-readable alert strings for overlays."""
    engine = AlertEngine()
    msg = engine.format_alert_message("eye_gaze", 0.85, "high")
    assert "Eye gaze look-away" in msg
    assert "0.85" in msg
    assert "HIGH" in msg


def test_event_lifecycle_and_debouncing():
    """Tests sequence of mock frames simulating a 5-second look-away event.

    Verifies that exactly 2 log lines are written (start + resolve),
    and that duration, peak confidence, and start/end timestamps are correct.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        logger = EventLogger(log_dir=temp_dir, batch_size=1, session_id="alert_test")
        engine = AlertEngine(event_logger=logger)

        # Sequence of frames over time:
        # Frame 0: t=100.0, triggered=True, confidence=0.5
        # Frame 1: t=101.0, triggered=True, confidence=0.8 (sustained, peak increases)
        # Frame 2: t=102.0, triggered=True, confidence=0.7 (sustained)
        # Frame 3: t=105.0, triggered=False, confidence=0.0 (resolved, duration=5.0)

        results_f0 = [
            DetectionResult(
                signal_name="eye_gaze",
                triggered=True,
                confidence=0.5,
                details={"gaze": "left"},
                timestamp=100.0
            )
        ]
        results_f1 = [
            DetectionResult(
                signal_name="eye_gaze",
                triggered=True,
                confidence=0.8,
                details={"gaze": "left"},
                timestamp=101.0
            )
        ]
        results_f2 = [
            DetectionResult(
                signal_name="eye_gaze",
                triggered=True,
                confidence=0.7,
                details={"gaze": "left"},
                timestamp=102.0
            )
        ]
        results_f3 = [
            DetectionResult(
                signal_name="eye_gaze",
                triggered=False,
                confidence=0.0,
                details={"gaze": "center"},
                timestamp=105.0
            )
        ]

        # Frame 0: Starts event
        alerts0 = engine.process(results_f0, frame_id=0)
        assert len(alerts0) == 1
        assert "Eye gaze look-away" in alerts0[0]

        # Frame 1: Active
        alerts1 = engine.process(results_f1, frame_id=1)
        assert len(alerts1) == 1

        # Frame 2: Active
        alerts2 = engine.process(results_f2, frame_id=2)
        assert len(alerts2) == 1

        # Frame 3: Resolves
        alerts3 = engine.process(results_f3, frame_id=3)
        assert len(alerts3) == 0

        # Ensure all writes are flushed to file
        logger.flush()

        # Check logs written
        assert os.path.exists(logger.log_path)
        with open(logger.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Exactly 2 log entries (start and resolved)
        assert len(lines) == 2

        # 1. Start Log Entry
        start_entry = json.loads(lines[0])
        assert start_entry["signal_name"] == "eye_gaze"
        assert start_entry["confidence"] == 0.5
        assert start_entry["frame_id"] == 0
        assert start_entry["details"]["status"] == "start"
        assert start_entry["details"]["severity"] == "medium"

        # 2. Resolved Log Entry
        resolved_entry = json.loads(lines[1])
        assert resolved_entry["signal_name"] == "eye_gaze"
        assert resolved_entry["confidence"] == 0.8  # Peak confidence
        assert resolved_entry["frame_id"] == 3
        assert resolved_entry["details"]["status"] == "resolved"
        assert resolved_entry["details"]["severity"] == "high"  # Severity from peak confidence
        assert resolved_entry["details"]["duration"] == 5.0
        assert resolved_entry["details"]["start_timestamp"] == 100.0
        assert resolved_entry["details"]["end_timestamp"] == 105.0
    finally:
        shutil.rmtree(temp_dir)
