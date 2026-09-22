# DRMS (Disaster Response Management System) - Fire & Smoke Intelligence Module

An intelligent aerial monitoring and decision-support system designed to detect wildfires, classify visual severity, track temporal kinematic growth trends, and compute automated triage urgency scores in real-time.

---

## Prerequisites & Installation

To run this project locally, make sure you have Python installed. Open your terminal and install the required dependencies using the following command:

```bash
python3 -m pip install fastapi uvicorn ultralytics opencv-python numpy scikit-learn

## How to Run the System
Step 1: Start the Backend Server
Run the FastAPI server using Uvicorn:
    uvicorn server:app --reload --port 8000
    
Step 2: Open the Frontend Dashboard
Open index.html in your web browser or go to:  http://localhost:8000

## How to Change the Test Video
Place your target video file inside your project directory (such as the test_videos/ folder).

Open server.py.

Locate the VIDEO_PATH variable near the top of the file:

Python
VIDEO_PATH = "test_videos/priority.mp4"
Simply change the filename string to your new video file (e.g., "test_videos/your_new_video.mp4").

Save the file and restart your FastAPI server.