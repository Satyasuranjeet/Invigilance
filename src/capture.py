import cv2
import numpy as np
import logging

class CameraNotFoundError(Exception):
    """Exception raised when the requested camera index cannot be opened."""
    pass

class VideoCapture:
    """Wrapper around cv2.VideoCapture to handle opening, reading, and downscaling frames.
    
    Known limitations:
    - Only handles default/indexed webcams, not complex IP streams.
    - Blocking read, tied to the camera's native FPS.
    - Relies on OpenCV default backend, which varies by OS.
    """
    def __init__(self, index: int = 0, max_width: int = 640):
        """Initializes the video capture device.
        
        Args:
            index (int): The index of the camera to open.
            max_width (int): The maximum width to downscale frames to (aspect ratio preserved).
        
        Raises:
            CameraNotFoundError: If the camera fails to open.
        """
        self.max_width = max_width
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise CameraNotFoundError(f"Failed to open camera at index {index}")
            
    def read_frame(self) -> np.ndarray | None:
        """Reads and optionally downscales the next frame from the camera.
        
        Returns:
            np.ndarray | None: The captured frame as a NumPy array, or None if read fails.
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
            
        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
        return frame
        
    def release(self) -> None:
        """Releases the camera resource."""
        self.cap.release()
