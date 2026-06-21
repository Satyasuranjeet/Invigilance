import os
import json
import shutil
import tempfile
from src.event_logger import EventLogger


def test_event_logger_initialization():
    """Tests that EventLogger initializes with correct log paths."""
    temp_dir = tempfile.mkdtemp()
    try:
        logger = EventLogger(log_dir=temp_dir, session_id="test_session")
        assert logger.log_dir == temp_dir
        assert logger.log_path == os.path.join(temp_dir, "session_test_session.jsonl")
        assert not os.path.exists(logger.log_path)
    finally:
        shutil.rmtree(temp_dir)


def test_event_logger_buffering():
    """Tests that EventLogger buffers writes and only writes on batch size limit or flush."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a logger with batch_size of 3
        logger = EventLogger(log_dir=temp_dir, batch_size=3, session_id="buffering_test")

        # Log 1st event
        logger.log_event("signal1", 0.5, {"x": 1}, 100.0, 1)
        assert len(logger.buffer) == 1
        assert not os.path.exists(logger.log_path)

        # Log 2nd event
        logger.log_event("signal2", 0.6, {"x": 2}, 101.0, 2)
        assert len(logger.buffer) == 2
        assert not os.path.exists(logger.log_path)

        # Log 3rd event (hits batch_size=3, should write to disk and clear buffer)
        logger.log_event("signal3", 0.7, {"x": 3}, 102.0, 3)
        assert len(logger.buffer) == 0
        assert os.path.exists(logger.log_path)

        # Verify written file content
        with open(logger.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3

        # Check first line schema
        first_entry = json.loads(lines[0])
        assert first_entry["signal_name"] == "signal1"
        assert first_entry["confidence"] == 0.5
        assert first_entry["details"] == {"x": 1}
        assert first_entry["timestamp"] == 100.0
        assert first_entry["frame_id"] == 1

        # Add 4th event (buffered)
        logger.log_event("signal4", 0.8, {"x": 4}, 103.0, 4)
        assert len(logger.buffer) == 1

        # Flush manually
        logger.flush()
        assert len(logger.buffer) == 0

        with open(logger.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 4
        assert json.loads(lines[3])["signal_name"] == "signal4"
    finally:
        shutil.rmtree(temp_dir)


def test_event_logger_close_and_destructor():
    """Tests that close flushes remaining events in buffer."""
    temp_dir = tempfile.mkdtemp()
    try:
        logger = EventLogger(log_dir=temp_dir, batch_size=5, session_id="cleanup_test")
        logger.log_event("signal", 0.9, {}, 200.0, 10)
        assert len(logger.buffer) == 1

        logger.close()
        assert len(logger.buffer) == 0
        assert os.path.exists(logger.log_path)

        with open(logger.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["signal_name"] == "signal"
    finally:
        shutil.rmtree(temp_dir)
