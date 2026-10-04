# RoadEye local inference console

This is a localhost-only FastAPI app for the trained sewer-defect checkpoints.
It supports image upload, background video processing, a browser webcam,
server-side USB cameras, and opt-in RTSP/HTTP IP-camera streams. It does not
upload images to a cloud service.

## Run on Windows

From the RoadEye project root, with the existing training venv:

```powershell
venv\Scripts\python.exe -m pip install -r dataset_crack\webapp\requirements.txt
venv\Scripts\python.exe -m uvicorn dataset_crack.webapp.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. OpenAPI documentation is at
<http://127.0.0.1:8000/docs>. Do not use `--reload` for normal inference:
restarts interrupt video jobs and can cause repeated GPU model loading.

For RTSP/HTTP cameras on a trusted local network, opt in before starting the
server:

```powershell
$env:ROADEYE_ALLOW_NETWORK_STREAMS = "1"
venv\Scripts\python.exe -m uvicorn dataset_crack.webapp.app:app --host 127.0.0.1 --port 8000
```

Do not bind this app to `0.0.0.0` or a public interface without adding
authentication, transport security, and upload/rate limits at a reverse proxy.
Network camera URLs can contain credentials; keep the browser and console
private. The server only accepts `rtsp`, `rtsps`, `http`, and `https` schemes.

## Modes

| Mode | Endpoint | Behavior |
|---|---|---|
| Image | `POST /api/infer/image?model_id=...&confidence=0.25` | Returns boxes, timing and an annotated JPG URL. Max 20 MB. |
| Video file | `POST /api/infer/video?model_id=...&stride=1` | Returns a job ID; poll `GET /api/jobs/{id}` and download `GET /api/jobs/{id}/output`. Max 2 GB. |
| Browser camera | `WS /ws/live/browser?model_id=...` | Browser sends JPEG bytes; server returns annotated JPEG/base64 plus boxes, FPS and model latency. |
| Server USB | `WS /ws/live/source?kind=usb&source=0&model_id=...` | Server opens camera index 0–8 via OpenCV. `GET /api/cameras` probes 0–3. |
| IP camera | `WS /ws/live/source?kind=network&source=rtsp://...&model_id=...` | Opt-in server-side RTSP/HTTP source. |
| Checkpoints | `GET /api/models` | Local availability, validation metrics, parameter counts and run provenance. |
| Health | `GET /api/health` | GPU/model status and network-stream setting. |

Only one checkpoint is kept loaded at a time to protect local VRAM. Switching
models incurs a loading pause, and simultaneous users share the inference
lock. Live FPS is measured end-to-end per delivered frame; image/video model
latency is shown separately. A faster model is generally preferable for live
feeds. USB and IP-camera capture could not be exercised without that hardware;
the browser-camera and source routes are implemented and can be tested locally.

On CUDA, RF-DETR uses in-place FP16 inference (`compile=False`) by default to
reduce latency and VRAM use. Set `ROADEYE_RFDETR_FP16=0` before starting the
server if you need unoptimized FP32 predictions. The saved validation scores
were produced with the training/evaluation setup, not this optional live
inference precision; small output differences are possible.

For videos, `stride=2` processes every second input frame and writes an output
at half the source FPS. The progress bar reports decoded source frames. OpenCV
creates an MP4, and if `ffmpeg` is installed the server transcodes to H.264 for
browser playback. The output is silent. If H.264 conversion is unavailable,
download the MP4 and play it in VLC if the browser does not support its codec.

## Models and scoring caveats

The selector contains checkpoints physically present under
`dataset_crack/runs/pipe_proto`, older `dataset_crack/runs/detect/runs/pipe_proto`,
and `dataset_crack/notebook/runs`. E3 is shown in the table but disabled because
its cloud-trained checkpoint is not local. Earlier or OOM-interrupted runs are
marked as such. The unrelated RoadEye pothole checkpoint is intentionally not
listed in this sewer-defect console.

CLAHE-trained models automatically apply the same Lab-L enhancement used to
build the v5 CLAHE dataset (clip limit 2.0, 8×8 grid) to incoming **raw**
frames. Do not upload already-CLAHE-processed imagery with those models.

Validation scores in the table come from `results.csv` (best validation row by
mAP50-95) or RF-DETR's `best_validation_metrics.json`. Test columns are filled
only where an unchanged test-split evaluation artifact exists; a dash means
there is no verified local test result. These scores are not all directly
comparable: v4/legacy/v6 use different data, and RF-DETR/YOLO have different
input sizes and evaluators. RF-DETR Nano's 90.65% is validation mAP50 after 50
epochs, **not** mAP50-95 or an untouched test-set result. Model-error-filtered
v6 scores must not be presented as independent test performance.

Parameter counts are measured from trusted local checkpoints in
`model_parameters.json`. After adding a checkpoint, restart the server and run:

```powershell
venv\Scripts\python.exe -m dataset_crack.webapp.inspect_parameters
```

This utility unpickles local YOLO `.pt` files. Never run it on untrusted model
files. RF-DETR `.pth` counts are read using safe tensor loading.

## Validation and storage

```powershell
venv\Scripts\python.exe -m unittest dataset_crack.webapp.test_app -v
venv\Scripts\python.exe -m dataset_crack.webapp.smoke_real
```

The first command uses mock inference to check all API flows. The second uses
the actual YOLO26s checkpoint and the bundled sample image/video. Uploaded
video and annotated outputs are kept under `dataset_crack/webapp/data/jobs/`;
the app intentionally does not automatically delete them. Jobs live in memory,
so a server restart makes old job URLs unavailable through the API even though
the files remain on disk. Review and clean that directory manually when needed.
