import asyncio
import cv2
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from classifier import classify_fire_crop
from temporal_classifier import TemporalGrowthClassifier
from priority_scorer import calculate_urgency_score

app = FastAPI()
model = YOLO("best.pt")
VIDEO_PATH = "test_videos/priority.mp4"

# Global storage for current telemetry state
latest_telemetry = {"frame": 0, "detections": []}

# Initialize the tracker globally
growth_tracker = TemporalGrowthClassifier(window_size=15)

def generate_frames():
    """Reads video, runs YOLO inference, evaluates severity, and yields JPEG frames for <img> tag."""
    global latest_telemetry
    
    # Run YOLO generator
    results = model.predict(
        source=VIDEO_PATH,
        imgsz=1024,
        conf=0.10,
        stream=True
    )

    for frame_idx, result in enumerate(results):
        detections = []
        original_frame = result.orig_img  # Get raw OpenCV frame for cropping
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detection_entry = {
                "class": class_name,
                "confidence": round(conf * 100, 1),
                "box": [x1, y1, x2, y2]
            }
            
            # Only run severity classification if the detected object is fire!
            if class_name.lower() == "fire":
                fire_crop = original_frame[y1:y2, x1:x2]
                severity_label, sev_conf = classify_fire_crop(fire_crop)
                detection_entry["severity"] = severity_label
                detection_entry["severity_confidence"] = round(sev_conf * 100, 1)
            
            detections.append(detection_entry)

        # Evaluate growth metrics for both fire and smoke after detections are populated
        growth_metrics = growth_tracker.update_and_evaluate(detections)
        max_priority = 0
        for det in detections:
            if det.get("class", "").lower() == "fire":
                p_score = calculate_urgency_score(det, growth_metrics["fire_behavior"])
                det["priority_score"] = p_score
                if p_score > max_priority:
                    max_priority = p_score

        # Update telemetry cache for WebSocket
        latest_telemetry = {
            "frame": frame_idx + 1,
            "detections": detections,
            "fire_behavior": growth_metrics["fire_behavior"],    # <--- Updated key
            "smoke_behavior": growth_metrics["smoke_behavior"],  # <--- Added smoke behavior key
            "max_urgency_score": max_priority
        }

        # Extract annotated frame as JPEG
        annotated_frame = result.plot()  # Draws bounding boxes on OpenCV frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # MJPEG stream frame chunk
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    """HTTP endpoint to stream annotated video frames."""
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint to push live JSON detection logs."""
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(latest_telemetry))
            await asyncio.sleep(0.05)  # 20 updates/sec
    except WebSocketDisconnect:
        pass

@app.post("/detect-severity")
async def detect_and_classify_fire(file: UploadFile = File(...)):
    """Endpoint for static image upload frame analysis."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image file."}

    # 1. Run primary object detection
    results = model(frame, verbose=False)
    detections = []
    
    # 2. Iterate through detected bounding boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        det_confidence = box.conf.item()
        
        # Crop the fire Region of Interest (ROI) from the frame
        fire_crop = frame[y1:y2, x1:x2]
        
        # 3. Pass crop to the secondary classification module
        severity_label, severity_confidence = classify_fire_crop(fire_crop)
        
        detections.append({
            "box": [x1, y1, x2, y2],
            "detection_confidence": round(det_confidence, 2),
            "severity_tier": severity_label,
            "severity_confidence": round(severity_confidence, 2),
        })
        
    return {"total_fires_detected": len(detections), "detections": detections}

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def read_index():
    """Serves the dashboard HTML directly from FastAPI on port 8000."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()