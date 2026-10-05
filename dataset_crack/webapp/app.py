"""FastAPI app for local RoadEye image, video and live pipe inspection."""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .engine import ModelManager, decode_image, encode_jpeg
from .registry import discover_models


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
JOBS_ROOT = APP_ROOT / "data" / "jobs"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOW_NETWORK_STREAMS = os.environ.get("ROADEYE_ALLOW_NETWORK_STREAMS", "0") == "1"

catalog = discover_models()
manager = ModelManager(catalog)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
source_locks: dict[str, threading.Lock] = {}
source_locks_guard = threading.Lock()
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="roadeye-video")

app = FastAPI(title="RoadEye Pipe Inspection", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


def _spec_or_404(model_id: str):
    spec = catalog.get(model_id)
    if spec is None:
        raise HTTPException(404, "Unknown model. Refresh the model list.")
    if not spec.checkpoint or not spec.checkpoint.is_file():
        raise HTTPException(409, "This model's checkpoint is not available locally.")
    return spec


def _update_job(job_id: str, **fields) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)


def _job_view(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found. It may belong to a previous server session.")
        return {key: value for key, value in job.items() if key not in {"input_path", "output_path"}}


def _open_usb(index: int):
    # Some Windows OpenCV builds expose DirectShow but cannot actually open it.
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY) if os.name == "nt" else (cv2.CAP_ANY,)
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture()


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_ROOT / "index.html", media_type="text/html")


@app.get("/api/health")
def health():
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        gpu = None
    return {
        "status": "ready", "models": sum(bool(m.checkpoint and m.checkpoint.is_file()) for m in catalog.values()),
        "gpu": gpu, "loaded_model": manager.current_id,
        "network_streams_enabled": ALLOW_NETWORK_STREAMS,
    }


@app.get("/api/models")
def models():
    return {"models": [item.public() for item in catalog.values()]}


@app.get("/api/cameras")
def cameras():
    """Probe a small set of server-side USB camera indexes."""
    found = []
    for index in range(4):
        cap = _open_usb(index)
        try:
            if cap.isOpened():
                found.append({"index": index, "label": f"USB camera {index}"})
        finally:
            cap.release()
    return {"cameras": found, "note": "These cameras are attached to the server computer."}


@app.post("/api/infer/image")
async def infer_image(
    file: UploadFile = File(...),
    model_id: str = Query(...),
    confidence: float = Query(0.25, ge=0.01, le=0.95),
):
    _spec_or_404(model_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(415, "Use a JPG, PNG, WebP or BMP image.")
    data = await file.read(MAX_IMAGE_BYTES + 1)
    await file.close()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the 20 MB limit.")
    try:
        frame = decode_image(data)
        started = time.perf_counter()
        result = await asyncio.to_thread(manager.infer, frame, model_id, confidence)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    job_id = uuid.uuid4().hex
    directory = JOBS_ROOT / job_id
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "annotated.jpg"
    output.write_bytes(encode_jpeg(result.frame, quality=90))
    elapsed = (time.perf_counter() - started) * 1000
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "kind": "image", "status": "complete", "progress": 1.0,
                        "output_path": output, "output_url": f"/api/jobs/{job_id}/output"}
    return {
        "job_id": job_id, "image_url": f"/api/jobs/{job_id}/output", "detections": result.detections,
        "inference_ms": round(result.inference_ms, 1), "model_load_ms": round(result.model_load_ms, 1),
        "total_ms": round(elapsed, 1), "width": frame.shape[1], "height": frame.shape[0],
    }


def _process_video(job_id: str, model_id: str, confidence: float, stride: int) -> None:
    with jobs_lock:
        input_path = jobs[job_id]["input_path"]
        directory = input_path.parent
    cap = cv2.VideoCapture(str(input_path))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError("Video could not be decoded. Try MP4/H.264 or AVI.")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if source_fps < 1 or source_fps > 240:
            source_fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 0 < width <= 7680 or not 0 < height <= 4320:
            raise ValueError("Unsupported video dimensions.")
        output = directory / "annotated.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps / stride, (width, height))
        if not writer.isOpened():
            raise RuntimeError("MP4 encoding is unavailable in this OpenCV build.")
        _update_job(job_id, status="processing", total_frames=total, width=width, height=height)
        started = time.perf_counter()
        seen = written = 0
        inference_sum = 0.0
        detection_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seen += 1
            if (seen - 1) % stride:
                continue
            result = manager.infer(frame, model_id, confidence)
            writer.write(result.frame)
            written += 1
            inference_sum += result.inference_ms
            detection_count += len(result.detections)
            if written == 1 or written % 5 == 0:
                elapsed = max(time.perf_counter() - started, 0.001)
                _update_job(
                    job_id, frames_processed=seen, frames_written=written,
                    progress=min(seen / total, 0.99) if total else None,
                    elapsed_s=round(elapsed, 1), processing_fps=round(written / elapsed, 2),
                    inference_ms_avg=round(inference_sum / written, 1),
                    detections=detection_count,
                )
        if not written:
            raise ValueError("The video contains no decodable frames.")
        writer.release()
        writer = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            _update_job(job_id, status="encoding", progress=0.99)
            browser_output = directory / "annotated_browser.mp4"
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(output),
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(browser_output),
            ]
            conversion = subprocess.run(command, capture_output=True, text=True, timeout=3600, check=False)
            if conversion.returncode == 0 and browser_output.is_file():
                output = browser_output
        _update_job(job_id, status="complete", progress=1.0, output_path=output,
                    output_url=f"/api/jobs/{job_id}/output", frames_processed=seen,
                    frames_written=written, processing_fps=round(written / max(time.perf_counter()-started, .001), 2),
                    inference_ms_avg=round(inference_sum / written, 1), detections=detection_count)
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))
    finally:
        cap.release()
        if writer is not None:
            writer.release()


@app.post("/api/infer/video", status_code=202)
async def infer_video(
    file: UploadFile = File(...),
    model_id: str = Query(...),
    confidence: float = Query(0.25, ge=0.01, le=0.95),
    stride: int = Query(1, ge=1, le=8),
):
    _spec_or_404(model_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(415, "Use MP4, AVI, MOV, MKV or WebM video.")
    job_id = uuid.uuid4().hex
    directory = JOBS_ROOT / job_id
    directory.mkdir(parents=True, exist_ok=False)
    input_path = directory / ("input" + suffix)
    size = 0
    try:
        with input_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(413, "Video exceeds the 2 GB limit.")
                output.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        directory.rmdir()
        raise
    finally:
        await file.close()
    if size == 0:
        raise HTTPException(400, "The uploaded video is empty.")
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "kind": "video", "status": "queued", "progress": 0.0,
                        "frames_processed": 0, "frames_written": 0, "total_frames": None,
                        "input_path": input_path, "output_path": None, "output_url": None,
                        "processing_fps": 0.0, "inference_ms_avg": None, "detections": 0}
    executor.submit(_process_video, job_id, model_id, confidence, stride)
    return _job_view(job_id)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job_view(job_id)


@app.get("/api/jobs/{job_id}/output")
def job_output(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job.get("status") != "complete" or not job.get("output_path"):
            raise HTTPException(404, "Output is not ready or this job is unknown.")
        output = job["output_path"]
    if not output.is_file():
        raise HTTPException(404, "Output file is missing.")
    return FileResponse(output, media_type="image/jpeg" if output.suffix == ".jpg" else "video/mp4",
                        headers={"Cache-Control": "no-store"})


async def _send_frame(ws: WebSocket, frame, model_id: str, confidence: float, fps: float):
    result = await asyncio.to_thread(manager.infer, frame, model_id, confidence)
    encoded = await asyncio.to_thread(encode_jpeg, result.frame)
    await ws.send_json({
        "type": "frame", "image": base64.b64encode(encoded).decode("ascii"),
        "detections": result.detections, "inference_ms": round(result.inference_ms, 1),
        "model_load_ms": round(result.model_load_ms, 1), "fps": round(fps, 1),
        "width": frame.shape[1], "height": frame.shape[0],
    })


@app.websocket("/ws/live/browser")
async def live_browser(ws: WebSocket, model_id: str, confidence: float = 0.25):
    await ws.accept()
    if model_id not in catalog or not catalog[model_id].checkpoint or not 0.01 <= confidence <= 0.95:
        await ws.send_json({"type": "error", "message": "Choose an available model and valid confidence."})
        await ws.close(code=1008)
        return
    previous = time.perf_counter()
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) > MAX_FRAME_BYTES:
                await ws.send_json({"type": "error", "message": "Camera frame is too large; reduce capture resolution."})
                continue
            try:
                frame = decode_image(data)
                now = time.perf_counter()
                fps = 1 / max(now - previous, 0.001)
                previous = now
                await _send_frame(ws, frame, model_id, confidence, fps)
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        return


def _source_key(kind: str, source: str) -> str:
    return f"{kind}:{source}"


def _source_lock(key: str) -> threading.Lock:
    with source_locks_guard:
        return source_locks.setdefault(key, threading.Lock())


@app.websocket("/ws/live/source")
async def live_source(
    ws: WebSocket, model_id: str, kind: str, source: str,
    confidence: float = 0.25, max_fps: int = 15,
):
    await ws.accept()
    if model_id not in catalog or not catalog[model_id].checkpoint or not 0.01 <= confidence <= 0.95:
        await ws.send_json({"type": "error", "message": "Choose an available model and valid confidence."})
        await ws.close(code=1008)
        return
    if not 1 <= max_fps <= 30:
        await ws.send_json({"type": "error", "message": "max_fps must be between 1 and 30."})
        await ws.close(code=1008)
        return
    if kind == "usb":
        if not source.isdecimal() or not 0 <= int(source) <= 8:
            await ws.send_json({"type": "error", "message": "USB camera index must be 0–8."})
            await ws.close(code=1008)
            return
        capture_source: int | str = int(source)
        backend = None
    elif kind == "network":
        parsed = urlparse(source)
        if not ALLOW_NETWORK_STREAMS or parsed.scheme not in {"rtsp", "rtsps", "http", "https"} or not parsed.hostname:
            await ws.send_json({"type": "error", "message": "Network streams are disabled or the URL is invalid. Set ROADEYE_ALLOW_NETWORK_STREAMS=1 on the local server."})
            await ws.close(code=1008)
            return
        capture_source = source
        backend = cv2.CAP_ANY
    else:
        await ws.send_json({"type": "error", "message": "Source must be usb or network."})
        await ws.close(code=1008)
        return
    lock = _source_lock(_source_key(kind, source))
    if not lock.acquire(blocking=False):
        await ws.send_json({"type": "error", "message": "That source is already open in another viewer."})
        await ws.close(code=1008)
        return
    cap = None
    try:
        cap = await asyncio.to_thread(_open_usb, capture_source) if kind == "usb" else await asyncio.to_thread(cv2.VideoCapture, capture_source, backend)
        if not cap.isOpened():
            await ws.send_json({"type": "error", "message": "Could not open the video source. Check its index or URL and camera permissions."})
            return
        previous = time.perf_counter()
        while True:
            cycle = time.perf_counter()
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok:
                await ws.send_json({"type": "error", "message": "The video source stopped sending frames."})
                break
            now = time.perf_counter()
            fps = 1 / max(now - previous, 0.001)
            previous = now
            await _send_frame(ws, frame, model_id, confidence, fps)
            await asyncio.sleep(max(0, 1 / max_fps - (time.perf_counter() - cycle)))
    except (WebSocketDisconnect, RuntimeError, FileNotFoundError):
        pass
    finally:
        if cap is not None:
            await asyncio.to_thread(cap.release)
        lock.release()
        try:
            await ws.close()
        except RuntimeError:
            pass
