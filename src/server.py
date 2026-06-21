import os
import sys
import json
import logging
import base64
from typing import Dict, List, Optional

# Add parent directory to sys.path to allow running this script directly
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np

from src.session_manager import ProctoringSession

logger = logging.getLogger("invigilance_server")

app = FastAPI(
    title="Invigilance Proctoring API Server",
    description="REST and WebSocket API for the real-time AI Exam-Proctoring system.",
    version="1.0.0"
)

# Enable CORS to allow direct browser requests from different origins (e.g. file:/// or dev servers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for active proctoring sessions
active_sessions: Dict[str, ProctoringSession] = {}


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "active_sessions_count": len(active_sessions)}


@app.post("/session/start")
def start_session(
    session_id: Optional[str] = Query(None, description="Custom session ID (optional)."),
    config_path: str = Query("config/thresholds.yaml", description="Path to configuration file."),
    model_path: str = Query("models/yolov8n.pt", description="Path to YOLOv8 model weights."),
    log_dir: str = Query("logs", description="Directory to store JSONL logs.")
):
    """Starts a new proctoring session and returns the session details."""
    session = ProctoringSession(
        session_id=session_id,
        config_path=config_path,
        model_path=model_path,
        log_dir=log_dir
    )
    
    sid = session.session_id
    if sid in active_sessions:
        raise HTTPException(status_code=400, detail=f"Session with ID '{sid}' is already active.")
        
    try:
        session.start()
        active_sessions[sid] = session
        return {
            "status": "started",
            "session_id": sid,
            "log_path": session.event_logger.log_path
        }
    except Exception as e:
        logger.error(f"Error starting session {sid}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not start session: {e}")


@app.post("/session/stop")
def stop_session(
    session_id: str = Query(..., description="The ID of the session to stop.")
):
    """Stops an active proctoring session and returns the final integrity report."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=444, detail=f"Active session with ID '{session_id}' not found.")
        
    session = active_sessions.pop(session_id)
    try:
        report = session.stop()
        return {
            "status": "stopped",
            "session_id": session_id,
            "report": report
        }
    except Exception as e:
        logger.error(f"Error stopping session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping session and generating report: {e}")


@app.post("/session/frame")
def process_frame_http(
    session_id: str = Query(..., description="Active session ID."),
    file: UploadFile = File(..., description="Webcam frame image file (JPEG/PNG).")
):
    """REST endpoint to upload a single frame and receive active alerts."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Active session with ID '{session_id}' not found.")
        
    session = active_sessions[session_id]
    try:
        file_bytes = file.file.read()
        active_alerts = session.process_encoded_frame(file_bytes)
        return {
            "session_id": session_id,
            "frame_id": session.frame_count - 1,
            "alerts": active_alerts
        }
    except Exception as e:
        logger.error(f"Error processing frame for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {e}")


@app.get("/session/active")
def list_active_sessions():
    """Lists currently active session IDs."""
    return {"active_sessions": list(active_sessions.keys())}


@app.websocket("/session/stream/{session_id}")
async def stream_frames_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint to stream frames from frontend and receive real-time alerts.

    Receives text messages (base64 JPEGs or JSON commands) or raw binary JPEGs.
    """
    # 1. Initialize session if not already registered via REST
    if session_id not in active_sessions:
        session = ProctoringSession(session_id=session_id)
        session.start()
        active_sessions[session_id] = session
    else:
        session = active_sessions[session_id]

    await websocket.accept()
    logger.info(f"WebSocket client connected for session {session_id}")

    try:
        while True:
            # Wait for data from frontend client
            message = await websocket.receive()
            
            # Stop signal or text message
            if "text" in message:
                text_data = message["text"]
                try:
                    # Check if it's a JSON command (e.g. stop command)
                    data_json = json.loads(text_data)
                    if data_json.get("type") == "stop":
                        # Stop session and return report
                        if session_id in active_sessions:
                            active_sessions.pop(session_id)
                        report = session.stop()
                        await websocket.send_json({
                            "type": "report",
                            "status": "stopped",
                            "report": report
                        })
                        logger.info(f"Session {session_id} stopped via WebSocket request.")
                        break
                except json.JSONDecodeError:
                    # If not JSON, check if it's a base64 encoded frame string
                    if text_data.startswith("data:image"):
                        # Extract raw base64 data
                        header, encoded = text_data.split(",", 1)
                        img_bytes = base64.b64decode(encoded)
                        alerts = session.process_encoded_frame(img_bytes)
                        await websocket.send_json({
                            "type": "alerts",
                            "frame_id": session.frame_count,
                            "alerts": alerts
                        })
                    
            elif "bytes" in message:
                # Raw binary JPEG image frame
                img_bytes = message["bytes"]
                alerts = session.process_encoded_frame(img_bytes)
                await websocket.send_json({
                    "type": "alerts",
                    "frame_id": session.frame_count,
                    "alerts": alerts
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
    finally:
        # Cleanup session on socket closure if it hasn't been stopped
        if session_id in active_sessions:
            active_sessions.pop(session_id)
            try:
                # Compile final report so we don't lose data
                report = session.stop()
                logger.info(f"Session {session_id} closed upon disconnect. Compiled report.")
            except Exception as e:
                logger.error(f"Failed to cleanly stop session {session_id} on disconnect: {e}")


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Helper function to run the FastAPI app via uvicorn programmatically."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
