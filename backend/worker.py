from contextlib import asynccontextmanager
import hmac

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(video_router)
app.include_router(stream_router)


@app.get("/health")
def health():
    return {"status": "ok", "device": settings.resolved_device()}


@app.middleware("http")
async def verify_inference_access(request: Request, call_next):
    if request.method != "OPTIONS" and (request.url.path.startswith("/infer/video") or request.url.path == "/auth/verify"):
        supplied = request.headers.get("authorization", "")
        token = supplied.removeprefix("Bearer ").strip() or request.query_params.get("access_token", "")
        if not settings.inference_token or not hmac.compare_digest(token, settings.inference_token):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing demo access code"})
    return await call_next(request)


@app.post("/auth/verify")
async def verify_access(request: Request):
    supplied = request.headers.get("authorization", "")
    token = supplied.removeprefix("Bearer ").strip()
    if not settings.inference_token or not hmac.compare_digest(token, settings.inference_token):
        return JSONResponse(status_code=401, content={"detail": "Invalid access code"})
    return {"valid": True}
