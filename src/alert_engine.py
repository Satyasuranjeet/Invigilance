import os
import yaml
from typing import List, Dict, Any, Optional
from src.detectors.base import DetectionResult
from src.event_logger import EventLogger


class AlertEngine:
    """Orchestrates detection results, applies debouncing, and logs events.

    Decides what counts as an event, logs when events start or resolve via
    the EventLogger, and maps signal confidence to severity tags.

    Known Limitations:
        - Relies on system clock time from DetectionResult timestamps.
        - Relies on thresholds config file existence for custom severities.
    """

    def __init__(self, config_path: str = "config/thresholds.yaml", event_logger: Optional[EventLogger] = None):
        """Initializes the AlertEngine.

        Args:
            config_path: Path to the thresholds YAML configuration file.
            event_logger: Optional EventLogger instance to write logs to.
        """
        self.config_path: str = config_path
        self.event_logger: Optional[EventLogger] = event_logger
        self.active_events: Dict[str, Dict[str, Any]] = {}
        self.severity_config: Dict[str, Any] = {}

        self._load_config()

    def _load_config(self) -> None:
        """Loads configuration from thresholds.yaml."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    self.severity_config = config.get("severity", {})
            except Exception as e:
                import sys
                print(f"Warning: Failed to load thresholds config: {e}. Using default severities.", file=sys.stderr)

    def get_severity(self, signal_name: str, confidence: float) -> str:
        """Determines the severity tag based on signal name and confidence.

        Args:
            signal_name: Name of the signal.
            confidence: Confidence score of the detection (0.0 to 1.0).

        Returns:
            One of 'low', 'medium', 'high'.
        """
        mapping = self.severity_config.get(signal_name, self.severity_config.get("default", {}))
        medium_thresh = mapping.get("medium", 0.4)
        high_thresh = mapping.get("high", 0.7)

        if confidence >= high_thresh:
            return "high"
        elif confidence >= medium_thresh:
            return "medium"
        return "low"

    def format_alert_message(self, signal_name: str, confidence: float, severity: str) -> str:
        """Formats a human-readable alert message for live overlay.

        Args:
            signal_name: Name of the signal.
            confidence: Confidence level.
            severity: Severity level string.

        Returns:
            Formatted alert string.
        """
        friendly_names = {
            "eye_gaze": "Eye gaze look-away",
            "head_pose": "Head turned away",
            "shoulder_pose": "Shoulder posture shift",
            "lip_movement": "Talking detected",
            "face_presence": "Face presence anomaly",
            "person_count": "Multiple or zero people in frame",
            "phone_detected": "Mobile phone visible",
            "secondary_device_detected": "Secondary recording device visible"
        }
        name = friendly_names.get(signal_name, signal_name.replace("_", " ").title())
        return f"{name} ({confidence:.2f} confidence) - {severity.upper()}"

    def process(self, results: List[DetectionResult], frame_id: int = 0) -> List[str]:
        """Processes detection results for a single frame.

        Applies event state transitions (start/resolved) and debounces alerts.

        Args:
            results: List of DetectionResult objects from active detectors.
            frame_id: The ID of the current video frame.

        Returns:
            List of human-readable alert strings for currently triggered detections.
        """
        alerts: List[str] = []

        for result in results:
            signal_name = result.signal_name
            severity = self.get_severity(signal_name, result.confidence)

            if result.triggered:
                # 1. Generate active alert string for overlay
                alerts.append(self.format_alert_message(signal_name, result.confidence, severity))

                # 2. Check for start transition
                if signal_name not in self.active_events:
                    self.active_events[signal_name] = {
                        "start_timestamp": result.timestamp,
                        "start_frame_id": frame_id,
                        "max_confidence": result.confidence,
                        "initial_details": result.details
                    }
                    if self.event_logger:
                        log_details = {
                            "status": "start",
                            "severity": severity,
                            "trigger_details": result.details
                        }
                        self.event_logger.log_event(
                            signal_name=signal_name,
                            confidence=result.confidence,
                            details=log_details,
                            timestamp=result.timestamp,
                            frame_id=frame_id
                        )
                else:
                    # Sustained event: update peak confidence seen so far
                    self.active_events[signal_name]["max_confidence"] = max(
                        self.active_events[signal_name]["max_confidence"],
                        result.confidence
                    )

            else:
                # 3. Check for resolved transition
                if signal_name in self.active_events:
                    event_info = self.active_events.pop(signal_name)
                    start_time = event_info["start_timestamp"]
                    duration = result.timestamp - start_time
                    peak_confidence = event_info["max_confidence"]
                    resolved_severity = self.get_severity(signal_name, peak_confidence)

                    if self.event_logger:
                        log_details = {
                            "status": "resolved",
                            "severity": resolved_severity,
                            "start_timestamp": start_time,
                            "end_timestamp": result.timestamp,
                            "duration": duration,
                            "trigger_details": event_info["initial_details"],
                            "resolve_details": result.details
                        }
                        self.event_logger.log_event(
                            signal_name=signal_name,
                            confidence=peak_confidence,
                            details=log_details,
                            timestamp=result.timestamp,
                            frame_id=frame_id
                        )

        return alerts
