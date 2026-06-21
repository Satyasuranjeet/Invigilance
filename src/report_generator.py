import os
import json
import logging
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)


def generate_report(
    log_source: Union[str, List[Dict[str, Any]]],
    session_id: str,
    start_time: float,
    end_time: float,
    total_frames: int
) -> Dict[str, Any]:
    """Generates a structured proctoring session report from events.

    Args:
        log_source: Either an absolute path to the session's JSONL log file,
                    or a list of already parsed event dictionaries.
        session_id: Unique identifier for the proctoring session.
        start_time: Epoch timestamp when the session started.
        end_time: Epoch timestamp when the session ended.
        total_frames: Total number of frames processed during the session.

    Returns:
        Dict[str, Any]: A complete structured proctoring session report.
    """
    events: List[Dict[str, Any]] = []

    # 1. Load events
    if isinstance(log_source, str):
        if os.path.exists(log_source):
            try:
                with open(log_source, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            events.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to read log file {log_source}: {e}")
        else:
            logger.warning(f"Log file not found at {log_source}")
    else:
        events = log_source

    duration_sec = max(0.0, end_time - start_time)
    avg_fps = float(total_frames / duration_sec) if duration_sec > 0 else 0.0

    # 2. Match start and resolved events to find duration periods
    anomaly_periods: List[Dict[str, Any]] = []
    # Temporary storage for active events that haven't been resolved yet
    active_starts: Dict[str, Dict[str, Any]] = {}

    for event in events:
        sig_name = event.get("signal_name")
        ts = event.get("timestamp")
        conf = event.get("confidence", 0.0)
        details = event.get("details", {})
        status = details.get("status")
        severity = details.get("severity", "low")

        if status == "start":
            active_starts[sig_name] = {
                "start_timestamp": ts,
                "start_frame_id": event.get("frame_id"),
                "max_confidence": conf,
                "severity": severity
            }
        elif status == "resolved":
            start_info = active_starts.pop(sig_name, None)
            if start_info:
                p_start = start_info["start_timestamp"]
                p_end = ts
                p_dur = max(0.0, p_end - p_start)
                max_conf = max(start_info["max_confidence"], conf)
                anomaly_periods.append({
                    "signal_name": sig_name,
                    "start_time": p_start,
                    "end_time": p_end,
                    "duration": p_dur,
                    "max_confidence": max_conf,
                    "severity": severity,
                    "resolved": True
                })
            else:
                # If we got a resolved event without a start event (e.g., logger initialized mid-event),
                # assume it started at the beginning of the session.
                p_start = start_time
                p_end = ts
                p_dur = max(0.0, p_end - p_start)
                anomaly_periods.append({
                    "signal_name": sig_name,
                    "start_time": p_start,
                    "end_time": p_end,
                    "duration": p_dur,
                    "max_confidence": conf,
                    "severity": severity,
                    "resolved": True
                })

    # Close any active events that remained unresolved at session termination
    for sig_name, start_info in active_starts.items():
        p_start = start_info["start_timestamp"]
        p_end = end_time
        p_dur = max(0.0, p_end - p_start)
        anomaly_periods.append({
            "signal_name": sig_name,
            "start_time": p_start,
            "end_time": p_end,
            "duration": p_dur,
            "max_confidence": start_info["max_confidence"],
            "severity": start_info["severity"],
            "resolved": False
        })

    # Sort anomaly periods by start time
    anomaly_periods.sort(key=lambda x: x["start_time"])

    # 3. Aggregate metrics by detector signal
    alert_counts: Dict[str, int] = {}
    alert_durations: Dict[str, float] = {}

    for period in anomaly_periods:
        sig = period["signal_name"]
        alert_counts[sig] = alert_counts.get(sig, 0) + 1
        alert_durations[sig] = alert_durations.get(sig, 0.0) + period["duration"]

    # 4. Calculate Integrity Score
    # Deduct points based on the duration of active alerts.
    # High-risk items deduct more points per second.
    penalty_multipliers = {
        "phone_detected": 4.0,           # High risk
        "secondary_device_detected": 3.0, # High risk
        "person_count": 2.0,             # Medium risk (multiple people or zero people)
        "face_presence": 1.5,            # Medium risk (no face or swapped face)
        "lip_movement": 1.5,             # Medium risk (talking/collusion)
        "eye_gaze": 0.8,                 # Low-medium risk
        "head_pose": 0.8,                # Low-medium risk
        "shoulder_pose": 0.4             # Low risk
    }

    total_penalty = 0.0
    for period in anomaly_periods:
        sig = period["signal_name"]
        mult = penalty_multipliers.get(sig, 0.5)
        dur = period["duration"]
        # Scale penalty by max confidence (i.e. more certain alerts deduct more)
        confidence_factor = period["max_confidence"]
        
        # Apply penalty: duration * multiplier * confidence
        total_penalty += dur * mult * confidence_factor

    integrity_score = max(0.0, min(100.0, 100.0 - total_penalty))

    # 5. Compile timeline events (high-severity alerts highlighted)
    friendly_names = {
        "eye_gaze": "Looking away from screen",
        "head_pose": "Head turned away",
        "shoulder_pose": "Shoulder posture shift",
        "lip_movement": "Talking / speaking",
        "face_presence": "Face missing / multiple faces",
        "person_count": "Multiple or zero people visible",
        "phone_detected": "Mobile phone visible",
        "secondary_device_detected": "Secondary recording device visible"
    }

    timeline = []
    for period in anomaly_periods:
        sig = period["signal_name"]
        action = friendly_names.get(sig, sig.replace("_", " ").title())
        status_str = "resolved" if period["resolved"] else "sustained until end"
        
        timeline.append({
            "timestamp": period["start_time"],
            "signal_name": sig,
            "severity": period["severity"].upper(),
            "duration_sec": round(period["duration"], 2),
            "max_confidence": round(period["max_confidence"], 2),
            "description": f"{action} detected for {period['duration']:.1f}s ({status_str})"
        })

    # Summary analysis
    suspicious_count = len([p for p in anomaly_periods if p["severity"] in ["medium", "high"]])
    
    if integrity_score >= 85:
        verdict = "PASS - High Integrity"
    elif integrity_score >= 60:
        verdict = "REVIEW - Moderate Integrity Anomalies"
    else:
        verdict = "FAIL - Suspicious Activity Flagged"

    report = {
        "session_id": session_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": round(duration_sec, 2),
        "total_frames": total_frames,
        "average_fps": round(avg_fps, 2),
        "integrity_score": round(integrity_score, 1),
        "verdict": verdict,
        "suspicious_events_count": suspicious_count,
        "alert_metrics": {
            sig: {
                "count": alert_counts.get(sig, 0),
                "total_duration_sec": round(alert_durations.get(sig, 0.0), 2)
            } for sig in alert_counts
        },
        "timeline": timeline
    }

    return report
