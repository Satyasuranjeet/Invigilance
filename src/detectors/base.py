from dataclasses import dataclass

@dataclass
class DetectionResult:
    signal_name: str
    triggered: bool
    confidence: float        # 0.0–1.0, never hardcode 1.0
    details: dict            # raw values (angles, ratios, bbox) for debugging
    timestamp: float
