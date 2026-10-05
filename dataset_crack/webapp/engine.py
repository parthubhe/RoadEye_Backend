"""Single-checkpoint inference engine shared by uploads and live sources."""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .registry import DATASET_ROOT, ModelSpec


CLASS_COLORS = [
    (68, 178, 248),   # crack
    (74, 122, 237),   # corrosion
    (73, 184, 142),   # sediment
    (145, 112, 236),  # roots
    (230, 167, 75),   # joints
]
MAX_DETECTIONS = 150


@dataclass
class InferenceResult:
    frame: np.ndarray
    detections: list[dict]
    inference_ms: float
    model_load_ms: float


def register_checkpoint_modules() -> None:
    """Resolve all project-defined classes before Ultralytics unpickles a local run."""
    config_root = DATASET_ROOT / "webapp" / "data" / "ultralytics"
    (config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    sys.path.insert(0, str(DATASET_ROOT))
    sys.path.insert(0, str(DATASET_ROOT / "notebook"))
    import ultralytics.nn.tasks as tasks
    from attention_modules import CBAM, ParallelAttention, SimAM, SqueezeExcitation
    from modules import DirectionalConv, WeightedFusion

    for cls in (CBAM, ParallelAttention, SimAM, SqueezeExcitation, DirectionalConv, WeightedFusion):
        setattr(tasks, cls.__name__, cls)
    # Older pickle paths reference these importable files directly.
    import simam  # noqa: F401
    import modules  # noqa: F401


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    scale = max(0.55, min(width, height) / 900)
    line = max(2, round(scale * 3))
    for item in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in item["xyxy"]]
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        color = CLASS_COLORS[item["class_id"] % len(CLASS_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line, cv2.LINE_AA)
        label = f"{item['class_name'].replace('_', ' ')} {item['confidence']:.2f}"
        text_scale = max(0.48, scale * 0.68)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
        top = max(0, y1 - text_height - baseline - 11)
        cv2.rectangle(out, (x1, top), (min(width - 1, x1 + text_width + 13), y1), color, -1)
        cv2.putText(out, label, (x1 + 6, max(text_height + 2, y1 - baseline - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (13, 26, 34), 2, cv2.LINE_AA)
    return out


def encode_jpeg(frame: np.ndarray, quality: int = 82) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode the annotated frame")
    return encoded.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("This image could not be decoded. Use JPG, PNG or WebP.")
    if frame.shape[0] * frame.shape[1] > 40_000_000:
        raise ValueError("Image exceeds the 40-megapixel safety limit.")
    return frame


class ModelManager:
    """Keep at most one model on the GPU; serialize inference across clients."""

    def __init__(self, catalog: dict[str, ModelSpec]):
        self.catalog = catalog
        self.lock = threading.RLock()
        self.current_id: str | None = None
        self.model = None

    def _unload(self) -> None:
        self.model = None
        self.current_id = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load(self, spec: ModelSpec) -> float:
        if self.current_id == spec.id and self.model is not None:
            return 0.0
        if not spec.checkpoint or not spec.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint unavailable for {spec.label}")
        self._unload()
        started = time.perf_counter()
        if spec.family == "RF-DETR Nano":
            import torch
            from rfdetr import RFDETRNano

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = RFDETRNano(
                pretrain_weights=str(spec.checkpoint), num_classes=5,
                resolution=spec.img_size, device=device,
            )
            if device == "cuda" and os.environ.get("ROADEYE_RFDETR_FP16", "1") == "1":
                # In-place casting avoids keeping two ~30M-parameter copies on
                # the laptop GPU; no TorchScript compile or warm-up graph.
                self.model.inference(compile=False, dtype=torch.float16, inplace=True)
        else:
            register_checkpoint_modules()
            from ultralytics import YOLO

            self.model = YOLO(str(spec.checkpoint))
        self.current_id = spec.id
        return (time.perf_counter() - started) * 1000

    def infer(self, frame: np.ndarray, model_id: str, confidence: float) -> InferenceResult:
        if model_id not in self.catalog:
            raise KeyError("Unknown model. Refresh the model list.")
        if not 0.01 <= confidence <= 0.95:
            raise ValueError("Confidence must be between 0.01 and 0.95")
        spec = self.catalog[model_id]
        if "CLAHE" in spec.dataset:
            from dataset_crack.build_v4_clahe import apply_clahe_bgr

            prepared_frame = apply_clahe_bgr(frame, clip_limit=2.0, tile_size=8)
        else:
            prepared_frame = frame
        with self.lock:
            load_ms = self._load(spec)
            started = time.perf_counter()
            if spec.family == "RF-DETR Nano":
                rgb = cv2.cvtColor(prepared_frame, cv2.COLOR_BGR2RGB)
                output = self.model.predict(Image.fromarray(rgb), threshold=confidence)
                if isinstance(output, list):
                    output = output[0]
                class_names = ["crack", "corrosion_rust", "sediment_deposit", "root_intrusion", "joint_defect"]
                detections = []
                for box, score, class_id in zip(output.xyxy, output.confidence, output.class_id):
                    class_id = int(class_id)
                    if class_id < 0 or class_id >= len(class_names):
                        continue
                    detections.append({
                        "xyxy": [float(v) for v in box], "confidence": float(score),
                        "class_id": class_id, "class_name": class_names[class_id],
                    })
            else:
                import torch

                device = 0 if torch.cuda.is_available() else "cpu"
                result = self.model.predict(
                    source=prepared_frame, imgsz=spec.img_size, conf=confidence,
                    device=device, verbose=False, max_det=MAX_DETECTIONS,
                )[0]
                names = result.names
                detections = []
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    detections.append({
                        "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                        "confidence": float(box.conf.item()),
                        "class_id": class_id,
                        "class_name": str(names.get(class_id, class_id)),
                    })
            inference_ms = (time.perf_counter() - started) * 1000
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        annotated = draw_detections(prepared_frame, detections)
        return InferenceResult(annotated, detections, inference_ms, load_ms)
