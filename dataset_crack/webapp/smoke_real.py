"""Optional end-to-end smoke test with real local checkpoints and sample media."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from .app import app, catalog
from .registry import DATASET_ROOT


def main() -> None:
    client = TestClient(app)
    model_id = next(identity for identity, spec in catalog.items() if spec.label == "YOLO26s · v5 baseline")
    image = DATASET_ROOT / "final_dataset_v5" / "valid" / "images" / "sewer_o8use_000004.jpg"
    with image.open("rb") as handle:
        result = client.post("/api/infer/image", params={"model_id": model_id},
                             files={"file": (image.name, handle, "image/jpeg")})
    result.raise_for_status()
    print("Image:", len(result.json()["detections"]), "detections,", result.json()["inference_ms"], "ms")

    video = DATASET_ROOT / "SEWER_SLAM_TEST.mp4"
    with video.open("rb") as handle:
        result = client.post("/api/infer/video", params={"model_id": model_id, "stride": 8},
                             files={"file": (video.name, handle, "video/mp4")})
    result.raise_for_status()
    job_id = result.json()["id"]
    for _ in range(120):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"complete", "error"}:
            break
        time.sleep(.5)
    print("Video:", job["status"], job.get("frames_written"), "frames,", job.get("processing_fps"), "FPS")
    if job["status"] != "complete":
        raise SystemExit(job.get("error", "Video job did not finish in 60 seconds"))
    output = client.get(job["output_url"])
    output.raise_for_status()
    print("Output:", output.headers["content-type"], len(output.content), "bytes")


if __name__ == "__main__":
    main()
