# QueueSense AI — Enhanced Website

## Features
- Responsive white/blue dashboard and sidebar navigation
- Local video upload and YOLO person detection through the existing FastAPI backend
- Annotated output video playback
- People count, peak count, prototype crowd level and baseline waiting-time estimate
- Analysis history loaded from the local JSONL history file
- Trend charts based on saved analysis runs
- CSV export of analysis history
- Configurable session alert threshold
- Basic Tamil/English interface toggle
- Mobile-friendly layout

## Run on Windows
Open Command Prompt in this folder (the folder containing `app.py` and `requirements.txt`). If your existing `.venv` is in the parent folder, run:

```cmd
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

If `.venv` is in this project folder instead, use:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser.

## Important limitations
- The pretrained YOLO model detects people in the whole video frame, not queue-specific zones.
- Waiting time is a simple baseline formula: average detected people × service minutes per person ÷ active counters. It is not a trained forecasting model.
- Crowd levels and alert thresholds are prototype settings, not official safety classifications.
- First-time YOLO model use may require internet access to download model weights unless `yolov8n.pt` is supplied locally.
- Use only video/CCTV sources you are authorized to access.
