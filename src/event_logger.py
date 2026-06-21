import os
import json
import time
import threading
from typing import List, Dict, Any, Optional


class EventLogger:
    """Handles structured event logging to a JSON Lines (JSONL) file.

    Logs are buffered in memory and written in batches to optimize disk I/O performance
    and minimize any FPS impact in the frame capture pipeline.

    Known Limitations:
        - If the process terminates abruptly (e.g., SIGKILL or power loss) without calling
          flush() or closing, any events currently buffered in memory will be lost.
        - Writing to disk is synchronous during flush, which could cause brief I/O blocking
          if the disk is highly loaded.
    """

    def __init__(self, log_dir: str = "logs", batch_size: int = 10, session_id: Optional[str] = None):
        """Initializes the EventLogger.

        Args:
            log_dir: The directory where logs should be written.
            batch_size: Number of log lines to buffer in memory before writing to disk.
            session_id: Optional string to use for the log filename. If not provided,
                        an epoch timestamp is used.
        """
        self.log_dir: str = log_dir
        self.batch_size: int = batch_size
        self._lock: threading.Lock = threading.Lock()
        self.buffer: List[Dict[str, Any]] = []

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        if not session_id:
            session_id = str(int(time.time()))
        self.log_path: str = os.path.join(self.log_dir, f"session_{session_id}.jsonl")

    def log_event(
        self,
        signal_name: str,
        confidence: float,
        details: Dict[str, Any],
        timestamp: float,
        frame_id: int
    ) -> None:
        """Buffers a single event for writing.

        Args:
            signal_name: Name of the detection signal (e.g., 'eye_gaze').
            confidence: Confidence score of the detection (0.0 to 1.0).
            details: Contextual details dictionary.
            timestamp: Epoch timestamp of the event.
            frame_id: The identifier of the frame where the event was detected.
        """
        entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "signal_name": signal_name,
            "confidence": confidence,
            "details": details,
            "frame_id": frame_id
        }

        with self._lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.batch_size:
                self._flush_unlocked()

    def flush(self) -> None:
        """Manually flushes any remaining logs in the buffer to disk."""
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Writes the buffered entries to disk. Caller must hold the lock."""
        if not self.buffer:
            return

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for entry in self.buffer:
                    f.write(json.dumps(entry) + "\n")
            self.buffer.clear()
        except IOError as e:
            # Handle potential writing errors without crashing the main application
            import sys
            print(f"Error writing to log file {self.log_path}: {e}", file=sys.stderr)

    def close(self) -> None:
        """Flushes remaining logs and performs cleanup."""
        self.flush()

    def __del__(self) -> None:
        """Ensures logs are flushed when the logger object is garbage collected."""
        try:
            self.flush()
        except Exception:
            pass
