import os
import time
import yaml
import logging
import numpy as np
from ultralytics import YOLO
from src.detectors.base import DetectionResult

logger = logging.getLogger(__name__)


class ObjectDetector:
    """YOLOv8n-based detector for person counting, mobile phone, and secondary recording device.

    Known limitations:
        - Accuracy is bounded by COCO-trained YOLOv8n's known weakness on small, partially occluded,
          or low-contrast objects (especially cell phones).
        - Secondary camera/device detection relies on a heuristic (cell phone or laptop presence)
          due to the lack of a dedicated 'camera' class in standard COCO. This can lead to false
          positives (e.g. background laptops) and false negatives (e.g. standalone DSLRs, webcams).
        - Detection is affected by lighting, camera resolution, motion blur, and angles.
    """

    def __init__(
        self,
        config_path: str = "config/thresholds.yaml",
        model_path: str = "models/yolov8n.pt",
    ) -> None:
        """Initializes the ObjectDetector with configuration thresholds and YOLO model.

        Args:
            config_path (str): Path to the YAML configuration file.
            model_path (str): Path to the YOLOv8n weight file.
        """
        # Default fallback values
        self.run_every_n_frames: int = 3
        self.confidence_threshold: float = 0.5

        # Load config
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config and "object_detection" in config:
                    obj_cfg = config["object_detection"]
                    self.run_every_n_frames = int(obj_cfg.get("run_every_n_frames", 3))
                    self.confidence_threshold = float(obj_cfg.get("confidence_threshold", 0.5))
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}. Using defaults.")
        else:
            logger.warning(f"Config file not found at {config_path}. Using defaults.")

        # Load YOLO model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            raise

        self.frame_count: int = 0
        self.last_results: list[DetectionResult] = []

    def analyze(self, frame: np.ndarray) -> list[DetectionResult]:
        """Analyzes a single video frame for people, cell phones, and laptops.

        Args:
            frame (np.ndarray): Input video frame in BGR format.

        Returns:
            list[DetectionResult]: List containing detection results for:
                - 'person_count'
                - 'phone_detected'
                - 'secondary_device_detected'
        """
        now = time.time()

        # Determine if we run inference or use cached results
        is_inference_frame = self.frame_count % self.run_every_n_frames == 0
        self.frame_count += 1

        if is_inference_frame or not self.last_results:
            # Perform inference
            try:
                # Run inference using the loaded YOLOv8n model
                # Use verbose=False to keep output clean and fast
                results = self.model(frame, conf=self.confidence_threshold, verbose=False, imgsz=480)

                boxes_data = results[0].boxes

                person_boxes: list[list[float]] = []
                phone_boxes: list[list[float]] = []
                laptop_boxes: list[list[float]] = []

                person_confs: list[float] = []
                phone_confs: list[float] = []
                laptop_confs: list[float] = []

                if boxes_data is not None:
                    clss = boxes_data.cls.cpu().numpy()
                    confs = boxes_data.conf.cpu().numpy()
                    xyxy = boxes_data.xyxy.cpu().numpy()

                    for cls_id, conf, box in zip(clss, confs, xyxy):
                        cls_id = int(cls_id)
                        conf = min(float(conf), 0.99)  # never hardcode 1.0
                        box_coords = [float(val) for val in box]

                        # 0 = person, 63 = laptop, 67 = cell phone
                        if cls_id == 0:
                            person_boxes.append(box_coords)
                            person_confs.append(conf)
                        elif cls_id == 67:
                            phone_boxes.append(box_coords)
                            phone_confs.append(conf)
                        elif cls_id == 63:
                            laptop_boxes.append(box_coords)
                            laptop_confs.append(conf)

                person_count = len(person_boxes)
                phone_count = len(phone_boxes)
                laptop_count = len(laptop_boxes)

                max_person_conf = max(person_confs) if person_confs else 0.0
                max_phone_conf = max(phone_confs) if phone_confs else 0.0
                max_laptop_conf = max(laptop_confs) if laptop_confs else 0.0

                # Signal 1: person_count (triggered if count != 1)
                person_triggered = person_count != 1
                person_result = DetectionResult(
                    signal_name="person_count",
                    triggered=person_triggered,
                    confidence=max_person_conf,
                    details={"count": person_count, "boxes": person_boxes, "stale": False},
                    timestamp=now,
                )

                # Signal 2: phone_detected (triggered if phone count > 0)
                phone_triggered = phone_count > 0
                phone_result = DetectionResult(
                    signal_name="phone_detected",
                    triggered=phone_triggered,
                    confidence=max_phone_conf,
                    details={"count": phone_count, "boxes": phone_boxes, "stale": False},
                    timestamp=now,
                )

                # Signal 3: secondary_device_detected (triggered if phone + laptop count > 0)
                sec_triggered = (phone_count + laptop_count) > 0
                max_sec_conf = max(max_phone_conf, max_laptop_conf)
                sec_result = DetectionResult(
                    signal_name="secondary_device_detected",
                    triggered=sec_triggered,
                    confidence=max_sec_conf,
                    details={
                        "phone_count": phone_count,
                        "laptop_count": laptop_count,
                        "boxes": phone_boxes + laptop_boxes,
                        "stale": False,
                    },
                    timestamp=now,
                )

                self.last_results = [person_result, phone_result, sec_result]

            except Exception as e:
                logger.error(f"YOLO inference failed: {e}")
                if not self.last_results:
                    self.last_results = self._create_default_results(now)
                else:
                    self._update_stale_results(now)
        else:
            # Re-use last results but update timestamp and set stale=True
            self._update_stale_results(now)

        return self.last_results

    def _update_stale_results(self, timestamp: float) -> None:
        """Helper to update the timestamp and mark details as stale for cached results."""
        updated = []
        for res in self.last_results:
            details_copy = dict(res.details)
            details_copy["stale"] = True
            updated.append(
                DetectionResult(
                    signal_name=res.signal_name,
                    triggered=res.triggered,
                    confidence=res.confidence,
                    details=details_copy,
                    timestamp=timestamp,
                )
            )
        self.last_results = updated

    def _create_default_results(self, timestamp: float) -> list[DetectionResult]:
        """Creates fallback/default results in case of initial error."""
        return [
            DetectionResult(
                signal_name="person_count",
                triggered=True,  # Safety default
                confidence=0.0,
                details={"count": 0, "boxes": [], "stale": False},
                timestamp=timestamp,
            ),
            DetectionResult(
                signal_name="phone_detected",
                triggered=False,
                confidence=0.0,
                details={"count": 0, "boxes": [], "stale": False},
                timestamp=timestamp,
            ),
            DetectionResult(
                signal_name="secondary_device_detected",
                triggered=False,
                confidence=0.0,
                details={"phone_count": 0, "laptop_count": 0, "boxes": [], "stale": False},
                timestamp=timestamp,
            ),
        ]
