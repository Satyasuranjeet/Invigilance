import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

# We import the app from src.server
from src.server import app, active_sessions

client = TestClient(app)


def test_health_endpoint():
    """Tests the /health REST endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("src.server.ProctoringSession")
def test_session_start_stop(mock_session_cls):
    """Tests starting and stopping a session via REST API."""
    # Reset active sessions
    active_sessions.clear()

    # Mock session instance behaviors
    mock_session = mock_session_cls.return_value
    mock_session.session_id = "test_rest_session_id"
    mock_session.event_logger = MagicMock()
    mock_session.event_logger.log_path = "logs/session_test_rest_session_id.jsonl"
    mock_session.stop.return_value = {"integrity_score": 95.0, "verdict": "PASS"}

    # 1. Start Session
    response = client.post("/session/start?session_id=test_rest_session_id")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "started"
    assert res_json["session_id"] == "test_rest_session_id"
    assert "test_rest_session_id" in active_sessions
    mock_session.start.assert_called_once()

    # 2. Stop Session
    response = client.post("/session/stop?session_id=test_rest_session_id")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["status"] == "stopped"
    assert res_json["report"]["integrity_score"] == 95.0
    assert "test_rest_session_id" not in active_sessions
    mock_session.stop.assert_called_once()


@patch("src.server.ProctoringSession")
def test_process_frame_endpoint(mock_session_cls):
    """Tests uploading a frame to the /session/frame endpoint."""
    active_sessions.clear()

    # Setup active mock session
    mock_session = mock_session_cls.return_value
    mock_session.session_id = "test_frame_session"
    mock_session.frame_count = 5
    mock_session.process_encoded_frame.return_value = ["Head turned away - MEDIUM"]
    
    active_sessions["test_frame_session"] = mock_session

    # Upload mock image bytes
    mock_img_bytes = b"fake-image-bytes-jpeg"
    response = client.post(
        "/session/frame?session_id=test_frame_session",
        files={"file": ("frame.jpg", mock_img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["session_id"] == "test_frame_session"
    assert res_json["alerts"] == ["Head turned away - MEDIUM"]
    mock_session.process_encoded_frame.assert_called_once_with(mock_img_bytes)


def test_stop_session_not_found():
    """Tests stopping a non-existent session yields 444 code."""
    response = client.post("/session/stop?session_id=invalid_id")
    assert response.status_code == 444
    assert "not found" in response.json()["detail"]
