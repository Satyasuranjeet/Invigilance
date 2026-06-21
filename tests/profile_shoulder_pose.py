import time
import cv2
from src.detectors.shoulder_pose import ShoulderPoseDetector

def profile():
    detector = ShoulderPoseDetector()
    frame = cv2.imread("tests/fixtures/portrait_straight.jpg")
    if frame is None:
        print("Failed to load fixture image")
        return
        
    # Warmup
    for _ in range(5):
        detector.analyze(frame)
        
    # Benchmark
    num_runs = 100
    start = time.perf_counter()
    for _ in range(num_runs):
        detector.analyze(frame)
    end = time.perf_counter()
    
    total_time = end - start
    avg_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time
    
    print(f"ShoulderPoseDetector Profile Results ({num_runs} frames):")
    print(f"  Average execution time: {avg_ms:.2f} ms/frame")
    print(f"  Estimated maximum speed: {fps:.2f} FPS")

if __name__ == "__main__":
    profile()
