from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.video import router as video_router
from app.api.stream import router as stream_router
from app.core import registry
from app.core.config import settings
from app.core.detector import YOLODetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = settings.resolved_device()
    detector = YOLODetector(settings.model_path, device=device)
    detector.load()
    registry.yolo_detector = detector
    print(f"[worker] Model loaded on {device}")
    yield
    registry.yolo_detector = None


app = FastAPI(title="PerimeterIQ Local Inference Worker", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https" + "://" + "perimeteriq.onrender.com", "http" + "://" + "localhost:5173", "http" + "://" + "127.0.0.1:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(video_router)
app.include_router(stream_router)


@app.get("/health")
def health():
    return {"status": "ok", "device": settings.resolved_device()}
