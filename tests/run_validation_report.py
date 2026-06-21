import os
import json
import time
import urllib.request
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# Import detectors and pipeline
from src.pipeline import ProctorPipeline
from src.detectors.base import DetectionResult

# Directories
FIXTURES_DIR = os.path.join("tests", "fixtures")
VAL_DIR = os.path.join(FIXTURES_DIR, "validation_set")
LOGS_DIR = "logs"

os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Unsplash Images for Validation Set
VALIDATION_IMAGES = {
    "portrait_straight.jpg": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640",
    "empty_desk.jpg": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=640",
    "two_people.jpg": "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=640",
    "phone_on_table.jpg": "https://images.unsplash.com/photo-1546054454-aa26e2b734c7?w=640",
    "laptop_only.jpg": "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=640"
}

# Define Validation Cases & Expected Triggers
VALIDATION_CASES = {
    "portrait_straight.jpg": {
        "expected_triggers": [],
        "description": "Standard portrait, looking straight with shoulders visible."
    },
    "empty_desk.jpg": {
        "expected_triggers": ["face_presence", "person_count"],
        "description": "Empty desk with 0 faces and 0 people."
    },
    "two_people.jpg": {
        "expected_triggers": ["face_presence", "person_count", "phone_detected", "secondary_device_detected"],
        "description": "Frame containing multiple faces, multiple people, and a phone."
    },
    "phone_on_table.jpg": {
        "expected_triggers": ["phone_detected", "secondary_device_detected", "face_presence", "person_count"],
        "description": "Cell phone visible on desk with no person present."
    },
    "laptop_only.jpg": {
        "expected_triggers": ["secondary_device_detected", "face_presence", "person_count"],
        "description": "Laptop visible on desk with no person present."
    },
    "simulated_eye_gaze_away.json": {
        "expected_triggers": ["eye_gaze"],
        "simulation": "eye_gaze_away",
        "description": "Gaze looking away continuously for 3.0s."
    },
    "simulated_head_yaw.json": {
        "expected_triggers": ["head_pose"],
        "simulation": "head_yaw",
        "description": "Head turned to the side exceeding 30 degrees yaw."
    },
    "simulated_shoulder_shift.json": {
        "expected_triggers": ["shoulder_pose"],
        "simulation": "shoulder_shift",
        "description": "Shoulder position shifted horizontally past 0.15 width."
    },
    "simulated_shoulder_low_visibility.json": {
        "expected_triggers": ["shoulder_pose"],
        "simulation": "shoulder_low_visibility",
        "description": "Shoulders missing or low visibility (< 0.5)."
    },
    "simulated_lip_movement.json": {
        "expected_triggers": ["lip_movement"],
        "simulation": "lip_movement",
        "description": "Simulated talking via oscillating mouth open/closed landmarks."
    },
    "simulated_normal_sequence.json": {
        "expected_triggers": [],
        "simulation": "normal_sequence",
        "description": "Stable straight gaze sequence verifying no false positives."
    },
    "simulated_look_away_brief.json": {
        "expected_triggers": [],
        "simulation": "look_away_brief",
        "description": "Brief 1.0s look away (below 2.0s duration threshold) - should not trigger."
    },
    "simulated_face_absence_brief.json": {
        "expected_triggers": [],
        "simulation": "face_absence_brief",
        "description": "Brief 1.0s face absence (below 1.5s duration threshold) - should not trigger."
    }
}

# Mock Classes for MediaPipe Results Injection
class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

class MockFace:
    def __init__(self, landmarks):
        self.landmark = landmarks

class MockFaceMeshResults:
    def __init__(self, faces):
        self.multi_face_landmarks = faces

class MockPoseLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class MockPoseLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

class MockPoseResults:
    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks

def download_file(filename: str, url: str) -> str:
    """Helper to download a validation image if it does not exist."""
    filepath = os.path.join(VAL_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        try:
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    return filepath

def get_base_landmarks(image_path: str):
    """Processes straight portrait once to get baseline landmarks for face-mesh simulations."""
    import mediapipe as mp
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    temp_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = temp_mesh.process(rgb)
    landmarks = []
    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks[0].landmark:
            landmarks.append((lm.x, lm.y, lm.z))
    temp_mesh.close()
    return landmarks

def get_base_pose_landmarks(image_path: str):
    """Processes straight portrait once to get baseline landmarks for pose simulations."""
    import mediapipe as mp
    img = cv2.imread(image_path)
    if img is None:
        return None
    temp_pose = mp.solutions.pose.Pose(model_complexity=0)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = temp_pose.process(rgb)
    landmarks = []
    if res.pose_landmarks:
        for lm in res.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z, lm.visibility))
    temp_pose.close()
    return landmarks

def main():
    print("=== Proctoring Tool Testing & Validation ===")
    
    # 1. Download missing validation images
    for filename, url in VALIDATION_IMAGES.items():
        download_file(filename, url)
        
    # 2. Write labels.json
    labels_path = os.path.join(VAL_DIR, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(VALIDATION_CASES, f, indent=2)
    print(f"Saved ground truth labels to {labels_path}")
    
    # 3. Load straight portrait for simulations
    straight_path = os.path.join(VAL_DIR, "portrait_straight.jpg")
    straight_img = cv2.imread(straight_path)
    if straight_img is None:
        print("Error: Could not load portrait_straight.jpg. Aborting validation.")
        return
        
    # Extract base landmarks for simulations
    base_landmarks = get_base_landmarks(straight_path)
    if base_landmarks is None or len(base_landmarks) == 0:
        print("Warning: Could not extract base landmarks. Using fallback.")
        base_landmarks = [(0.5, 0.5, 0.0) for _ in range(478)]
        
    base_pose_landmarks = get_base_pose_landmarks(straight_path)
    if base_pose_landmarks is None or len(base_pose_landmarks) == 0:
        print("Warning: Could not extract base pose landmarks. Using fallback.")
        base_pose_landmarks = [(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
        
    # Signals we track
    SIGNALS = [
        "eye_gaze",
        "head_pose",
        "shoulder_pose",
        "lip_movement",
        "face_presence",
        "person_count",
        "phone_detected",
        "secondary_device_detected"
    ]
    
    # Initialize metrics structure: signal_name -> {TP, FP, TN, FN}
    metrics = {sig: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for sig in SIGNALS}
    
    # Iterate through all validation cases
    case_results = []
    for filename, case_info in VALIDATION_CASES.items():
        print(f"\nEvaluating: {filename} ({case_info['description']})")
        expected = case_info["expected_triggers"]
        simulation = case_info.get("simulation", None)
        
        # Instantiate a fresh pipeline per case to prevent state leak
        pipeline = ProctorPipeline(config_path="config/thresholds.yaml", model_path="models/yolov8n.pt")
        
        # We run 30 frames at 10 FPS (simulated)
        num_frames = 35 if simulation == "shoulder_shift" else 30
        triggered_signals = set()
        
        start_sim_time = 1000.0
        
        for frame_idx in range(num_frames):
            sim_time = start_sim_time + (frame_idx * 0.1) # 10 FPS
            
            # Prepare frame and mocks
            frame_img = straight_img.copy() if simulation else cv2.imread(os.path.join(VAL_DIR, filename))
            mock_mesh_results = None
            mock_pose_results = None
            
            if simulation == "eye_gaze_away":
                # Shift iris landmarks 468 and 473 horizontally to simulate looking away
                lm_list = [MockLandmark(x, y, z) for (x, y, z) in base_landmarks]
                lm_list[468].x += 0.05
                lm_list[473].x += 0.05
                mock_mesh_results = MockFaceMeshResults([MockFace(lm_list)])
                
            elif simulation == "head_yaw":
                # Shift nose (1) and chin (152) horizontally to simulate head turn
                lm_list = [MockLandmark(x, y, z) for (x, y, z) in base_landmarks]
                lm_list[1].x -= 0.1
                lm_list[152].x -= 0.1
                mock_mesh_results = MockFaceMeshResults([MockFace(lm_list)])
                
            elif simulation == "shoulder_shift":
                # Frames 0-29: baseline center at ~0.5
                # Frames 30-34: shifted center by +0.20
                lm_list = [MockPoseLandmark(x, y, z, vis) for (x, y, z, vis) in base_pose_landmarks]
                if frame_idx >= 30:
                    lm_list[11].x += 0.20
                    lm_list[12].x += 0.20
                mock_pose_results = MockPoseResults(MockPoseLandmarks(lm_list))
                
            elif simulation == "shoulder_low_visibility":
                # Set LEFT_SHOULDER (11) and RIGHT_SHOULDER (12) visibility to 0.1
                lm_list = [MockPoseLandmark(x, y, z, vis) for (x, y, z, vis) in base_pose_landmarks]
                lm_list[11].visibility = 0.1
                lm_list[12].visibility = 0.1
                mock_pose_results = MockPoseResults(MockPoseLandmarks(lm_list))
                
            elif simulation == "lip_movement":
                # Alternate MAR (mouth aspect ratio)
                lm_list = [MockLandmark(x, y, z) for (x, y, z) in base_landmarks]
                if frame_idx % 2 == 0:
                    # Open mouth
                    lm_list[13].y = base_landmarks[13][1] - 0.04
                    lm_list[14].y = base_landmarks[14][1] + 0.04
                else:
                    # Closed mouth
                    lm_list[13].y = base_landmarks[13][1]
                    lm_list[14].y = base_landmarks[14][1]
                mock_mesh_results = MockFaceMeshResults([MockFace(lm_list)])
                
            elif simulation == "look_away_brief":
                # Frames 10-19: look away (1.0s duration - below 2.0s duration threshold)
                lm_list = [MockLandmark(x, y, z) for (x, y, z) in base_landmarks]
                if 10 <= frame_idx < 20:
                    lm_list[468].x += 0.05
                    lm_list[473].x += 0.05
                mock_mesh_results = MockFaceMeshResults([MockFace(lm_list)])
                
            elif simulation == "face_absence_brief":
                # Frames 10-19: face absent (1.0s duration - below 1.5s duration threshold)
                # Keep person in frame so YOLO doesn't trigger person_count, but return zero faces in mesh
                if 10 <= frame_idx < 20:
                    mock_mesh_results = MockFaceMeshResults(None)
                else:
                    lm_list = [MockLandmark(x, y, z) for (x, y, z) in base_landmarks]
                    mock_mesh_results = MockFaceMeshResults([MockFace(lm_list)])
                    
            elif simulation == "normal_sequence":
                lm_list = [MockLandmark(x, y, z) for (x, y, z) in base_landmarks]
                mock_mesh_results = MockFaceMeshResults([MockFace(lm_list)])
                
            if frame_img is None:
                print(f"Error loading frame {frame_idx} for case {filename}")
                continue
                
            # Process frame with patched time and mock results if provided
            with patch("time.time", return_value=sim_time):
                if mock_mesh_results is not None and mock_pose_results is not None:
                    with patch.object(pipeline.shared_face_mesh, "process", return_value=mock_mesh_results):
                        with patch.object(pipeline.shoulder_pose_detector.pose, "process", return_value=mock_pose_results):
                            results = pipeline.process_frame(frame_img)
                elif mock_mesh_results is not None:
                    with patch.object(pipeline.shared_face_mesh, "process", return_value=mock_mesh_results):
                        results = pipeline.process_frame(frame_img)
                elif mock_pose_results is not None:
                    with patch.object(pipeline.shoulder_pose_detector.pose, "process", return_value=mock_pose_results):
                        results = pipeline.process_frame(frame_img)
                else:
                    results = pipeline.process_frame(frame_img)
                    
            # Check if any signal triggers
            for res in results:
                if res.triggered:
                    triggered_signals.add(res.signal_name)
                    
        # Log case results
        actual_list = sorted(list(triggered_signals))
        print(f"  Expected Triggers: {expected}")
        print(f"  Actual Triggers:   {actual_list}")
        
        case_results.append({
            "filename": filename,
            "description": case_info["description"],
            "expected_triggers": expected,
            "actual_triggers": actual_list
        })
        
        # Calculate metric counts for each signal
        for sig in SIGNALS:
            exp_trig = sig in expected
            act_trig = sig in triggered_signals
            
            if exp_trig and act_trig:
                metrics[sig]["TP"] += 1
            elif not exp_trig and act_trig:
                metrics[sig]["FP"] += 1
            elif exp_trig and not act_trig:
                metrics[sig]["FN"] += 1
            else:
                metrics[sig]["TN"] += 1

    # 4. Generate Final Metrics Table
    print("\n=== Validation Summary Report ===")
    
    summary_report = {}
    
    # Console print formatting
    print(f"{'Signal Name':<30} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'Precision':<10} | {'Recall':<10} | {'FPR':<10}")
    print("-" * 92)
    
    for sig in SIGNALS:
        tp = metrics[sig]["TP"]
        fp = metrics[sig]["FP"]
        tn = metrics[sig]["TN"]
        fn = metrics[sig]["FN"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        summary_report[sig] = {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "fpr": float(round(fpr, 4))
        }
        
        print(f"{sig:<30} | {tp:<4} | {fp:<4} | {tn:<4} | {fn:<4} | {precision:<10.4f} | {recall:<10.4f} | {fpr:<10.4f}")

    # Save to logs/validation_report.json
    output_report = {
        "timestamp": time.time(),
        "cases_evaluated": len(VALIDATION_CASES),
        "results": case_results,
        "metrics": summary_report
    }
    
    report_file_path = os.path.join(LOGS_DIR, "validation_report.json")
    with open(report_file_path, "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"\nSaved structured JSON report to {report_file_path}")
    print("========================================")

if __name__ == "__main__":
    main()
