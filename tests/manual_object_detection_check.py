import os
import time
import urllib.request
import cv2
from src.detectors.object_detector import ObjectDetector

# Create fixtures directory if not exists
FIXTURES_DIR = os.path.join("tests", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

# URL maps for test images
IMAGE_URLS = {
    "empty_desk.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", # Or an actual empty room image
    "phone_on_table.jpg": "https://images.unsplash.com/photo-1546054454-aa26e2b734c7?w=640",
    "two_people.jpg": "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=640"
}

# Wait, the bus.jpg has persons but no desk, let's find a real empty desk or empty room image that has no person:
# Let's search/download a reliable empty desk image:
IMAGE_URLS["empty_desk.jpg"] = "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=640"


def download_fixtures() -> None:
    """Downloads sample images for manual validation if not already present."""
    for filename, url in IMAGE_URLS.items():
        filepath = os.path.join(FIXTURES_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from {url}...")
            try:
                # Add headers to avoid HTTP 403 Forbidden from some domains
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"Saved to {filepath}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")


def run_checks() -> None:
    """Runs YOLOv8n object detection on three sample images and prints results."""
    download_fixtures()
    
    print("\nInitializing ObjectDetector...")
    detector = ObjectDetector()
    
    for filename in IMAGE_URLS.keys():
        filepath = os.path.join(FIXTURES_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping check for {filename} (not downloaded)")
            continue
            
        print(f"\n========================================")
        print(f"Testing image: {filename}")
        print(f"========================================")
        
        frame = cv2.imread(filepath)
        if frame is None:
            print(f"Failed to read image {filepath}")
            continue
            
        # Ensure we run inference (reset frame count so first image runs inference)
        detector.frame_count = 0
        
        start_time = time.perf_counter()
        results = detector.analyze(frame)
        duration = (time.perf_counter() - start_time) * 1000.0
        
        print(f"Analysis completed in {duration:.2f} ms")
        for res in results:
            print(f"- Signal: {res.signal_name}")
            print(f"  Triggered: {res.triggered}")
            print(f"  Confidence: {res.confidence:.4f}")
            print(f"  Stale: {res.details.get('stale')}")
            # Keep details print simple
            details_summary = {
                k: v for k, v in res.details.items() if k != "boxes"
            }
            details_summary["box_count"] = len(res.details.get("boxes", []))
            print(f"  Details: {details_summary}")


def profile_performance() -> None:
    """Profiles the ObjectDetector analyze speed over 50 iterations."""
    filepath = os.path.join(FIXTURES_DIR, "phone_on_table.jpg")
    if not os.path.exists(filepath):
        print("Phone image not available. Skipping profiling.")
        return
        
    frame = cv2.imread(filepath)
    if frame is None:
        print("Failed to load image for profiling.")
        return
        
    print("\n========================================")
    print("Profiling Performance (50 frames)...")
    print("========================================")
    
    detector = ObjectDetector()
    
    # We want to measure actual inference performance, but we must account for run_every_n_frames.
    # To measure YOLO inference time specifically, we force frame_count % run_every_n_frames == 0,
    # or we can run it sequentially and measure the overall time including skipped frames.
    # Let's measure two things:
    # 1. Direct YOLO inference time (by forcing frame_count to run inference)
    # 2. Average pipeline frame processing time (where 1 in 3 frames does inference)
    
    # 1. Direct YOLO inference time
    yolo_times = []
    for _ in range(50):
        detector.frame_count = 0  # Force inference
        start = time.perf_counter()
        detector.analyze(frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        yolo_times.append(elapsed)
        
    avg_yolo = sum(yolo_times) / len(yolo_times)
    print(f"Average direct YOLO inference time: {avg_yolo:.2f} ms/frame")
    
    # 2. Pipeline processing time (with frame skipping, e.g. every 3rd frame)
    detector.frame_count = 0
    pipeline_times = []
    for _ in range(50):
        start = time.perf_counter()
        detector.analyze(frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        pipeline_times.append(elapsed)
        
    avg_pipeline = sum(pipeline_times) / len(pipeline_times)
    print(f"Average pipeline processing time (run_every_3rd_frame): {avg_pipeline:.2f} ms/frame")
    print(f"Estimated pipeline FPS: {1000.0 / avg_pipeline:.2f} FPS")


if __name__ == "__main__":
    run_checks()
    profile_performance()
