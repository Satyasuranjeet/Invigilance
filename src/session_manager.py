import os
import time
import logging
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from src.pipeline import ProctorPipeline
from src.alert_engine import AlertEngine
from src.event_logger import EventLogger
from src.report_generator import generate_report

logger = logging.getLogger(__name__)


class ProctoringSession:
    """Manages the full lifecycle of a proctoring session.

    This SDK class wraps the pipeline execution, event logging, alert engine,
    and report compilation. It can be instantiated directly by Python clients
    or an API server.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        config_path: str = "config/thresholds.yaml",
        model_path: str = "models/yolov8n.pt",
        log_dir: str = "logs"
    ) -> None:
        """Initializes the ProctoringSession.

        Args:
            session_id: Optional custom session identifier. If None, uses epoch timestamp.
            config_path: Path to thresholds.yaml.
            model_path: Path to YOLOv8n.pt / YOLOv8n.onnx.
            log_dir: Directory where logs should be stored.
        """
        self.session_id: str = session_id or str(int(time.time()))
        self.config_path: str = config_path
        self.model_path: str = model_path
        self.log_dir: str = log_dir

        self.pipeline: Optional[ProctorPipeline] = None
        self.alert_engine: Optional[AlertEngine] = None
        self.event_logger: Optional[EventLogger] = None

        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.frame_count: int = 0
        self.active: bool = False

    def start(self) -> None:
        """Starts the proctoring session, initializing models and logging."""
        if self.active:
            logger.warning("Session is already active.")
            return

        self.start_time = time.time()
        self.frame_count = 0
        self.active = True

        # Initialize core components
        self.event_logger = EventLogger(
            log_dir=self.log_dir,
            batch_size=5,
            session_id=self.session_id
        )
        self.alert_engine = AlertEngine(
            config_path=self.config_path,
            event_logger=self.event_logger
        )
        self.pipeline = ProctorPipeline(
            config_path=self.config_path,
            model_path=self.model_path
        )

        logger.info(f"Proctoring session {self.session_id} successfully started.")

    def process_frame(self, frame: np.ndarray) -> List[str]:
        """Processes a single frame and returns active alerts.

        Args:
            frame: A BGR OpenCV image array.

        Returns:
            List[str]: A list of human-readable alert messages active on this frame.
        """
        if not self.active:
            raise RuntimeError("Session has not been started yet. Call start() first.")

        h, w = frame.shape[:2]

        # Downscale frame to max 640px width for inference (performance budget rule)
        inference_width = 640
        if w > inference_width:
            scale = float(inference_width) / w
            inference_frame = cv2.resize(frame, (inference_width, int(h * scale)))
        else:
            inference_frame = frame.copy()

        # Run detection pipeline
        results = self.pipeline.process_frame(inference_frame)
        
        # Debounce and log events
        active_alerts = self.alert_engine.process(results, frame_id=self.frame_count)
        
        self.frame_count += 1
        return active_alerts

    def process_encoded_frame(self, image_bytes: bytes) -> List[str]:
        """Decodes an image from raw bytes (e.g. JPEG) and processes it.

        Args:
            image_bytes: Raw bytes of the image (JPEG, PNG, etc.)

        Returns:
            List[str]: Active alert messages.
        """
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode frame bytes.")
            return []
        return self.process_frame(frame)

    def stop(self) -> Dict[str, Any]:
        """Stops the session, closes logs, and compiles the integrity report.

        Returns:
            Dict[str, Any]: The final session report.
        """
        if not self.active:
            raise RuntimeError("Session is not active.")

        self.end_time = time.time()
        self.active = False

        # Close and flush log buffer
        log_path = self.event_logger.log_path
        self.event_logger.close()

        logger.info(f"Proctoring session {self.session_id} stopped. Generating report...")

        # Generate structured report from the JSONL log file
        report = generate_report(
            log_source=log_path,
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=self.end_time,
            total_frames=self.frame_count
        )

        return report
