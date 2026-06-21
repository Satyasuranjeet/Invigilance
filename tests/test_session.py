import os
import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.report_generator import generate_report
from src.session_manager import ProctoringSession


def test_generate_report_basic():
    """Tests the report generator with a basic sequence of mock events."""
    start_time = 1000.0
    end_time = 1050.0  # 50 seconds duration
    total_frames = 500

    # Mock list of JSONL events:
    # 1. eye_gaze start at 1010, resolve at 1020 (duration 10s)
    # 2. phone_detected start at 1030, stays unresolved (duration 20s until end_time)
    mock_events = [
        {
            "timestamp": 1010.0,
            "signal_name": "eye_gaze",
            "confidence": 0.8,
            "details": {"status": "start", "severity": "medium"},
            "frame_id": 100
        },
        {
            "timestamp": 1020.0,
            "signal_name": "eye_gaze",
            "confidence": 0.8,
            "details": {
                "status": "resolved",
                "severity": "medium",
                "start_timestamp": 1010.0,
                "end_timestamp": 1020.0,
                "duration": 10.0
            },
            "frame_id": 200
        },
        {
            "timestamp": 1030.0,
            "signal_name": "phone_detected",
            "confidence": 0.9,
            "details": {"status": "start", "severity": "high"},
            "frame_id": 300
        }
    ]

    report = generate_report(
        log_source=mock_events,
        session_id="test_session_123",
        start_time=start_time,
        end_time=end_time,
        total_frames=total_frames
    )

    assert report["session_id"] == "test_session_123"
    assert report["duration_sec"] == 50.0
    assert report["total_frames"] == 500
    assert report["average_fps"] == 10.0
    assert report["suspicious_events_count"] == 2  # both medium and high count

    # Check alert metrics
    assert "eye_gaze" in report["alert_metrics"]
    assert report["alert_metrics"]["eye_gaze"]["count"] == 1
    assert report["alert_metrics"]["eye_gaze"]["total_duration_sec"] == 10.0

    assert "phone_detected" in report["alert_metrics"]
    assert report["alert_metrics"]["phone_detected"]["count"] == 1
    assert report["alert_metrics"]["phone_detected"]["total_duration_sec"] == 20.0  # 1050 - 1030 = 20s

    # Check integrity score calculation
    # Penalty eye_gaze: 10s * 0.8 * 0.8 (eye_gaze penalty is 0.8) = 6.4
    # Penalty phone_detected: 20s * 4.0 * 0.9 (phone penalty is 4.0) = 72.0
    # Total Penalty = 78.4 -> Score = 21.6
    assert abs(report["integrity_score"] - 21.6) < 0.1
    assert report["verdict"] == "FAIL - Suspicious Activity Flagged"

    # Check timeline
    assert len(report["timeline"]) == 2
    assert report["timeline"][0]["signal_name"] == "eye_gaze"
    assert report["timeline"][1]["signal_name"] == "phone_detected"


@patch("src.session_manager.EventLogger")
@patch("src.session_manager.AlertEngine")
@patch("src.session_manager.ProctorPipeline")
def test_proctoring_session_lifecycle(mock_pipeline_cls, mock_alert_engine_cls, mock_event_logger_cls):
    """Tests the ProctoringSession start, process, and stop lifecycle."""
    session = ProctoringSession(session_id="test_lifecycle_session", log_dir="temp_logs")
    
    # Assert initial state
    assert session.active is False
    assert session.frame_count == 0

    # Start session
    session.start()
    assert session.active is True
    mock_event_logger_cls.assert_called_once()
    mock_alert_engine_cls.assert_called_once()
    mock_pipeline_cls.assert_called_once()

    # Process frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.process_frame.return_value = []
    
    mock_alert_engine = mock_alert_engine_cls.return_value
    mock_alert_engine.process.return_value = ["Mock Alert message"]

    alerts = session.process_frame(mock_frame)
    assert alerts == ["Mock Alert message"]
    assert session.frame_count == 1

    # Mock log path for generate_report mock call
    session.event_logger.log_path = "temp_logs/session_test_lifecycle_session.jsonl"

    # Stop session
    with patch("src.session_manager.generate_report") as mock_gen_report:
        mock_gen_report.return_value = {"integrity_score": 100.0}
        
        report = session.stop()
        
        assert session.active is False
        session.event_logger.close.assert_called_once()
        mock_gen_report.assert_called_once()
        assert report == {"integrity_score": 100.0}
