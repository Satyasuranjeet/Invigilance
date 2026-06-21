import time
import cv2
from src.pipeline import ProctorPipeline

def profile():
    print("Initializing full proctoring pipeline for profiling...")
    pipeline = ProctorPipeline()
    
    # Load sample portrait image
    frame = cv2.imread("tests/fixtures/portrait_straight.jpg")
    if frame is None:
        print("Failed to load fixture image")
        return
        
    # Downscale frame to 640px max width just like main.py
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640.0 / w
        frame_input = cv2.resize(frame, (640, int(h * scale)))
    else:
        frame_input = frame.copy()
        
    print(f"Profiling input frame dimensions: {frame_input.shape[1]}x{frame_input.shape[0]}")
    
    # Warmup
    print("Warming up pipeline models...")
    for _ in range(5):
        pipeline.process_frame(frame_input)
        
    # Benchmark
    num_runs = 50
    print(f"Benchmarking pipeline over {num_runs} frames...")
    start = time.perf_counter()
    for i in range(num_runs):
        pipeline.process_frame(frame_input)
    end = time.perf_counter()
    
    total_time = end - start
    avg_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time
    
    print(f"\nProctorPipeline Profile Results ({num_runs} frames):")
    print(f"  Total processing time: {total_time:.3f} s")
    print(f"  Average execution time per frame: {avg_ms:.2f} ms/frame")
    print(f"  Estimated maximum speed: {fps:.2f} FPS")

if __name__ == "__main__":
    profile()
