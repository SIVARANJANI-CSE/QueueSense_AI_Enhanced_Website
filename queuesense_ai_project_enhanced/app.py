from pathlib import Path
from datetime import datetime
import json, shutil, uuid
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
STATIC, UPLOADS, OUTPUTS = BASE/"static", BASE/"uploads", BASE/"outputs"
HISTORY, WEIGHTS = BASE/"history.jsonl", BASE/"yolov8n.pt"
for folder in (STATIC, UPLOADS, OUTPUTS): folder.mkdir(exist_ok=True)
app = FastAPI(title="QueueSense AI")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")
model = None

def get_model():
    global model
    if model is None:
        try: model = YOLO(str(WEIGHTS) if WEIGHTS.exists() else "yolov8n.pt")
        except Exception as exc: raise HTTPException(500, f"Could not load YOLO. Install dependencies and model weights first. Details: {exc}")
    return model

def crowd_label(count):
    if count < 5: return "Low"
    if count < 15: return "Medium"
    if count < 30: return "High"
    return "Critical"

@app.get("/")
def index(): return FileResponse(STATIC/"index.html")

@app.get("/api/health")
def health(): return {"status":"ok", "weights_available":WEIGHTS.exists()}

@app.get("/api/history")
def get_history():
    if not HISTORY.exists(): return {"items":[]}
    rows=[]
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    return {"items":list(reversed(rows[-25:]))}

@app.post("/api/analyze")
async def analyze(video: UploadFile=File(...), counters:int=Form(2), service_minutes:float=Form(2.0), confidence:float=Form(.35), every_n_frames:int=Form(5)):
    if not video.filename: raise HTTPException(400,"Choose a video file.")
    suffix=Path(video.filename).suffix.lower()
    if suffix not in {".mp4",".avi",".mov",".mkv",".webm"}: raise HTTPException(400,"Use MP4, AVI, MOV, MKV or WEBM video.")
    if counters < 1 or service_minutes <= 0: raise HTTPException(400,"Counters must be at least 1 and service time must be positive.")
    job=uuid.uuid4().hex[:10]; source=UPLOADS/f"{job}{suffix}"; rendered=OUTPUTS/f"{job}_annotated.mp4"
    try:
        with source.open("wb") as dest: shutil.copyfileobj(video.file,dest)
        cap=cv2.VideoCapture(str(source))
        if not cap.isOpened(): raise HTTPException(400,"Could not open video. Try a standard MP4.")
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        writer=cv2.VideoWriter(str(rendered),cv2.VideoWriter_fourcc(*"mp4v"),max(1.0,fps),(width,height))
        if not writer.isOpened():
            cap.release(); raise HTTPException(500,"Could not create annotated video.")
        detector=get_model(); counts=[]; boxes_cache=[]; frame_num=0
        stride=max(1,min(30,int(every_n_frames))); conf=min(.95,max(.05,float(confidence)))
        while True:
            ok,frame=cap.read()
            if not ok: break
            if frame_num % stride == 0:
                result=detector.predict(frame,classes=[0],conf=conf,verbose=False)[0]
                boxes_cache=result.boxes.xyxy.cpu().numpy().astype(int).tolist() if result.boxes is not None else []
                counts.append(len(boxes_cache))
            for coords in boxes_cache:
                x1,y1,x2,y2=coords
                cv2.rectangle(frame,(x1,y1),(x2,y2),(40,175,105),2)
                cv2.putText(frame,"Person",(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,175,105),2)
            cv2.rectangle(frame,(8,8),(315,48),(22,32,55),-1)
            cv2.putText(frame,f"People detected: {counts[-1] if counts else 0}",(18,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            writer.write(frame); frame_num+=1
        cap.release(); writer.release()
        if not counts: raise HTTPException(400,"No readable video frames found.")
        average=round(sum(counts)/len(counts),1); peak=max(counts)
        wait=round(average*float(service_minutes)/int(counters),1); level=crowd_label(average)
        advice=("Crowd is elevated. Consider opening another counter or redirecting visitors if safe and staffing permits." if level in ("High","Critical") else "Monitor the queue and prepare extra service capacity during peak periods." if level=="Medium" else "Crowd level is currently low. Continue monitoring.")
        row={"id":job,"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"filename":video.filename,"average_people":average,"peak_people":peak,"crowd_level":level,"waiting_minutes":wait,"counters":counters,"sampled_frames":len(counts)}
        with HISTORY.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\\n")
        return {**row,"video_url":f"/outputs/{rendered.name}","recommendation":advice,"note":"Prototype estimate: average detected people × service minutes per person ÷ active counters. Not a trained ML forecast; full-frame count may include people not in the queue."}
    finally:
        await video.close()
        try:
            if source.exists(): source.unlink()
        except OSError: pass
