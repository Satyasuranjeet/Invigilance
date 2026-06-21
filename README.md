# Invigilance — Real-time AI Proctoring System

**Invigilance** is a lightweight, real-time exam-proctoring tool designed to run locally on a standard laptop webcam without requiring GPU acceleration. The system monitors a candidate's video feed frame-by-frame, runs high-frequency CPU-friendly inference, and logs structured event alerts when anomalous behaviors are detected.

---

## 🚀 Key Features & Detectors

Invigilance processes frames through a unified orchestrator pipeline to track 8 distinct proctoring signals:

1. **Eye Gaze Tracking (`eye_gaze`)**: Estimates iris center deviation relative to inner and outer eye corners using MediaPipe Face Mesh iris refinement. Detects looking away from the screen for sustained periods (> 2.0s).
2. **Head Pose Estimation (`head_pose`)**: Maps 3D head rotation (yaw/pitch/roll) using facial landmarks and OpenCV's `solvePnP` solver to track looking down, sideways, or away.
3. **Shoulder Posture Tracking (`shoulder_pose`)**: Uses MediaPipe Pose to track left and right shoulders, flagging lateral shifts (leaning out of frame) or low shoulder visibility.
4. **Lip Movement & Talking Detection (`lip_movement`)**: Computes Mouth Aspect Ratio (MAR) variance and tracks rolling mean-crossing oscillations. This filters out static open-mouth shapes (like yawning) and triggers exclusively on talking.
5. **Face Presence Verification (`face_presence`)**: Counts faces in the frame. Generates alerts if zero faces (sustained for > 1.5s) or multiple faces are detected.
6. **Person Detection (`person_count`)**: Leverages a fast YOLOv8n nano object detection model to assert that exactly one person is present in the frame.
7. **Mobile Phone Detection (`phone_detected`)**: Detects cell phones visible in the camera frame with high confidence using YOLOv8n.
8. **Secondary Screen Detection (`secondary_device_detected`)**: Uses object detection heuristics to flag secondary screens or laptops present in the workspace.

---

## 🛠️ Locked Tech Stack

* **Core Language**: Python 3.11+
* **Face & Body Landmarks**: MediaPipe (`Face Mesh` with iris refinement, `Pose` model complexity 0) — optimized for CPU-only, sub-30ms execution.
* **Object Detection**: `Ultralytics YOLOv8n` (nano) model exported to ONNX for fast inference.
* **Video I/O & Rendering**: OpenCV (`cv2`)
* **Math & Geometry**: NumPy
* **Configuration**: YAML (`PyYAML`) for threshold and severity configuration.
* **Logging**: Structured JSON-lines (`.jsonl`) event logger.

---

## 📦 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Satyasuranjeet/Invigilance.git
   cd Invigilance
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8n Model Weights**:
   Ensure you run the downloader script to pull the weights required for object detection:
   ```bash
   python download_model.py
   ```

---

## 🖥️ Running the Application

### 1. Live Monitoring HUD (Webcam)
Launch the main application to monitor your camera stream in real-time:
```bash
python main.py
```
* **Real-time Diagnostic Console (Bottom-Left)**: Displays numeric metrics such as gaze angle, head yaw/pitch, shoulder shifts, and mouth MAR variance.
* **Futuristic Bounding Boxes**: Overlays bounding boxes on faces, persons, phones, and laptops.
* **Pulsing Status Panel (Top-Left)**: Displays connection and pipeline speed in Frames Per Second (FPS).
* **Alert Engine Stack (Top-Right)**: Shows active notifications color-coded by severity level (Yellow: Low, Orange: Medium, Red: High) after debouncing.
* *Press **`q`** to exit the application gracefully.*

---

## 🧪 Testing & Validation

### 1. Execute Unit Tests
Verify functional correctness and model integrations:
```bash
pytest
```

### 2. Run Validation Report
Evaluate the accuracy metrics (Precision, Recall, FPR) of all detectors against a standardized test set consisting of static validation images and temporal mock sequences:
```bash
# On Windows (PowerShell):
$env:PYTHONPATH="."
python tests/run_validation_report.py
```
This generates a console summary and saves a structured JSON validation report to `logs/validation_report.json`.

---

## ⚙️ Configuration

Thresholds, temporal durations, frame rates, and severity levels are configured inside `config/thresholds.yaml`. 

```yaml
eye_gaze:
  away_angle_deg: 25
  away_duration_sec: 2.0
head_pose:
  yaw_threshold_deg: 30
  pitch_threshold_deg: 20
shoulder:
  lateral_shift_ratio: 0.15
lip:
  movement_var_threshold: 0.02
  min_crossings: 3
object_detection:
  run_every_n_frames: 3
  confidence_threshold: 0.5
face_presence:
  absence_duration_sec: 1.5
```
For a detailed analysis of capabilities, performance figures, and edge-case limitations, refer to `LIMITATIONS.md`.
