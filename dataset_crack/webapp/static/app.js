const $ = (id) => document.getElementById(id);
const state = { models: [], mode: "image", socket: null, cameraStream: null, pollTimer: null, busy: false, networkEnabled: false };

function pct(value) { return value == null ? "—" : `${(value * 100).toFixed(2)}%`; }
function params(value) { return value == null ? "—" : `${(value / 1e6).toFixed(2)} M`; }
function selectedModel() { return state.models.find((m) => m.id === $("modelSelect").value); }
function confidence() { return Number($("confidence").value); }
function setStatus(text, status = "") {
  const node = $("systemStatus");
  node.className = `system-status ${status}`;
  node.querySelector("span:last-child").textContent = text;
}
function error(message) {
  const node = $("errorMessage");
  node.textContent = message;
  node.classList.remove("hidden");
}
function clearError() { $("errorMessage").classList.add("hidden"); }
function progress(label, percent, detail = "") {
  $("progressWrap").classList.remove("hidden");
  $("progressLabel").textContent = label;
  $("progressDetail").textContent = detail;
  if (percent == null) {
    $("progressBar").removeAttribute("value");
    $("progressPercent").textContent = "Working";
  } else {
    $("progressBar").value = Math.max(0, Math.min(100, percent));
    $("progressPercent").textContent = `${Math.round(percent)}%`;
  }
}
function hideProgress() { $("progressWrap").classList.add("hidden"); }
function setBusy(busy) {
  state.busy = busy;
  $("runImage").disabled = busy;
  $("runVideo").disabled = busy;
}
function setView(kind, source, caption, badge = "Result") {
  if (kind !== "video") $("videoResult").pause();
  for (const id of ["viewportEmpty", "imageResult", "videoResult", "liveResult"]) $(id).classList.add("hidden");
  if (kind === "image") { $("imageResult").src = source; $("imageResult").classList.remove("hidden"); }
  else if (kind === "video") { $("videoResult").src = source; $("videoResult").classList.remove("hidden"); $("videoResult").load(); }
  else if (kind === "live") { $("liveResult").src = source; $("liveResult").classList.remove("hidden"); }
  else $("viewportEmpty").classList.remove("hidden");
  $("viewCaption").textContent = caption;
  $("viewState").textContent = badge;
  $("viewState").classList.toggle("active", kind !== "empty");
}
function telemetry(ms, fps, count, label) {
  $("inferenceMs").textContent = ms == null ? "—" : Number(ms).toFixed(1);
  $("inferenceFps").textContent = fps == null ? "—" : Number(fps).toFixed(1);
  $("detectionCount").textContent = count == null ? "—" : String(count);
  $("activeModel").textContent = label || "—";
}
function setDownload(url, filename = "result") {
  const link = $("downloadResult");
  link.classList.toggle("hidden", !url);
  if (url) { link.href = url; link.download = filename; }
}
function switchMode(mode) {
  if (state.socket) stopLive();
  clearError();
  state.mode = mode;
  for (const name of ["image", "video", "live"]) {
    $(`${name}Panel`).classList.toggle("hidden", name !== mode);
    const tab = document.querySelector(`[data-mode="${name}"]`);
    tab.classList.toggle("active", name === mode);
    tab.setAttribute("aria-selected", String(name === mode));
  }
  hideProgress();
}
function addCell(row, text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  row.appendChild(cell);
  return cell;
}
function renderModels() {
  const tbody = $("modelsTable");
  tbody.replaceChildren();
  const ordered = [...state.models].sort((a, b) => Number(b.available) - Number(a.available) || (b.metrics.map50 ?? -1) - (a.metrics.map50 ?? -1));
  for (const model of ordered) {
    const row = document.createElement("tr");
    row.title = model.notes || model.run || "";
    addCell(row, model.label);
    addCell(row, model.dataset);
    const status = addCell(row, model.available ? model.status : "Unavailable");
    status.className = "";
    const pill = document.createElement("span");
    pill.textContent = status.textContent;
    pill.className = `state-pill ${model.status === "complete" && model.available ? "" : "dim"}`;
    status.replaceChildren(pill);
    addCell(row, params(model.parameters), "metric");
    addCell(row, pct(model.metrics.map50), "metric");
    addCell(row, pct(model.metrics.map5095), "metric");
    addCell(row, `${pct(model.metrics.precision)} / ${pct(model.metrics.recall)}`, "metric");
    addCell(row, pct(model.test_metrics.map50), "metric");
    addCell(row, pct(model.test_metrics.map5095), "metric");
    addCell(row, `${pct(model.test_metrics.precision)} / ${pct(model.test_metrics.recall)}`, "metric");
    tbody.appendChild(row);
  }
  $("tableModelCount").textContent = `${state.models.length} recorded runs`;
  const rfRuns = state.models.filter(m => m.family === "RF-DETR Nano" && m.status === "complete" && m.test_metrics.map50 != null);
  const rf40 = rfRuns.find(m => m.label.includes("40 epochs"));
  const rf50 = rfRuns.find(m => m.label.includes("50 epochs"));
  const rfPartial = state.models.find(m => m.family === "RF-DETR Nano" && m.status !== "complete" && m.test_metrics.map50 != null);
  $("rfdetrTestNote").textContent = rf40 && rf50
    ? `Held-out v5 test (1,986 images): RF-DETR Nano 40 epochs ${pct(rf40.test_metrics.map50)} mAP50 / ${pct(rf40.test_metrics.map5095)} mAP50-95; 50 epochs ${pct(rf50.test_metrics.map50)} mAP50 / ${pct(rf50.test_metrics.map5095)} mAP50-95.${rfPartial ? ` Earlier interrupted checkpoint ${pct(rfPartial.test_metrics.map50)} mAP50 / ${pct(rfPartial.test_metrics.map5095)} mAP50-95.` : ""} These are test results, separate from validation scores.`
    : "RF-DETR held-out test results are not available for both completed checkpoints yet; blank table cells mean no verified test artifact.";
  $("modelCount").textContent = `${state.models.filter(m => m.available).length} local checkpoints`;
  const select = $("modelSelect");
  select.replaceChildren();
  for (const model of state.models.filter(m => m.available)) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label;
    select.appendChild(option);
  }
  const preferred = state.models.find(m => m.label === "YOLO26s · v5 baseline" && m.available);
  if (preferred) select.value = preferred.id;
  updateModelHint();
}
function updateModelHint() {
  const model = selectedModel();
  $("modelHint").textContent = model ? `${model.dataset} · ${model.parameters == null ? "parameter count unavailable" : params(model.parameters) + " parameters"}${model.notes ? " · " + model.notes : ""}` : "No local checkpoints found.";
  $("activeModel").textContent = model ? model.label : "—";
}
async function apiError(response) {
  try { const payload = await response.json(); return payload.detail || payload.message || `Request failed (${response.status}).`; }
  catch { return `Request failed (${response.status}).`; }
}
function upload(url, file, onUpload, onComplete, onError) {
  const xhr = new XMLHttpRequest();
  xhr.open("POST", url);
  xhr.upload.onprogress = (event) => { if (event.lengthComputable) onUpload(event.loaded / event.total); };
  xhr.onerror = () => onError("Network connection to the local server was lost.");
  xhr.onload = () => {
    let payload;
    try { payload = JSON.parse(xhr.responseText); } catch { payload = {}; }
    if (xhr.status >= 200 && xhr.status < 300) onComplete(payload);
    else onError(payload.detail || `Upload failed (${xhr.status}).`);
  };
  const form = new FormData();
  form.append("file", file);
  xhr.send(form);
}
function inspectImage() {
  const file = $("imageFile").files[0], model = selectedModel();
  if (!file || !model) { error("Choose an image and an available checkpoint first."); return; }
  if (state.busy) return;
  clearError(); setBusy(true); setDownload(null);
  progress("Uploading image", 0, file.name);
  const query = new URLSearchParams({ model_id: model.id, confidence: confidence() });
  upload(`/api/infer/image?${query}`, file,
    fraction => progress(fraction >= 1 ? "Running model" : "Uploading image", fraction >= 1 ? null : fraction * 85,
      fraction >= 1 ? "First use of a checkpoint can take several seconds." : file.name),
    data => {
      setBusy(false); hideProgress();
      setView("image", `${data.image_url}?t=${Date.now()}`, `${file.name} · ${data.width} × ${data.height}`, "Inspected");
      telemetry(data.inference_ms, null, data.detections.length, model.label);
      $("resultSummary").textContent = `${data.detections.length} detections · ${data.model_load_ms ? `model load ${data.model_load_ms.toFixed(0)} ms · ` : ""}inference ${data.inference_ms.toFixed(1)} ms. Review the boxes before use.`;
      setDownload(data.image_url, "roadeye_annotated.jpg");
    },
    message => { setBusy(false); hideProgress(); error(message); }
  );
}
async function pollVideo(jobId, model, filename) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    if (!response.ok) throw new Error(await apiError(response));
    const job = await response.json();
    if (job.status === "error") throw new Error(job.error || "Video processing failed.");
    if (job.status === "complete") {
      clearTimeout(state.pollTimer); state.pollTimer = null; setBusy(false);
      progress("Complete", 100, `${job.frames_written} annotated frames`);
      setTimeout(hideProgress, 1500);
      setView("video", job.output_url, `${filename} · ${job.frames_written} frames written`, "Ready to play");
      telemetry(job.inference_ms_avg, job.processing_fps, job.detections, model.label);
      $("resultSummary").textContent = `Processed ${job.frames_written} frames at ${job.processing_fps} output FPS. Download the annotated MP4 if playback is unsupported by your browser.`;
      setDownload(job.output_url, "roadeye_annotated.mp4");
      return;
    }
    const fraction = job.progress;
    progress(job.status === "queued" ? "Waiting for worker" : job.status === "encoding" ? "Encoding browser video" : "Processing video", fraction == null ? null : 10 + fraction * 90,
      job.total_frames ? `${job.frames_processed} / ${job.total_frames} source frames · ${job.processing_fps || 0} output FPS` : `${job.frames_written} frames written`);
    if (job.frames_written) telemetry(job.inference_ms_avg, job.processing_fps, job.detections, model.label);
    state.pollTimer = setTimeout(() => pollVideo(jobId, model, filename), 800);
  } catch (cause) { setBusy(false); hideProgress(); error(cause.message); }
}
function inspectVideo() {
  const file = $("videoFile").files[0], model = selectedModel();
  if (!file || !model) { error("Choose a video and an available checkpoint first."); return; }
  if (state.busy) return;
  clearError(); setBusy(true); setDownload(null);
  progress("Uploading video", 0, file.name);
  const query = new URLSearchParams({ model_id: model.id, confidence: confidence(), stride: $("stride").value });
  upload(`/api/infer/video?${query}`, file,
    fraction => progress("Uploading video", fraction * 10, `${(fraction * 100).toFixed(0)}% transferred`),
    data => { progress("Queued", 10, "Preparing the video decoder."); pollVideo(data.id, model, file.name); },
    message => { setBusy(false); hideProgress(); error(message); }
  );
}
function liveUrl(path, query) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}${path}?${new URLSearchParams(query)}`;
}
function stopLive() {
  if (state.socket) { state.socket.close(); state.socket = null; }
  if (state.cameraStream) { state.cameraStream.getTracks().forEach(track => track.stop()); state.cameraStream = null; }
  $("browserCamera").srcObject = null;
  $("startLive").classList.remove("hidden");
  $("stopLive").classList.add("hidden");
  $("viewState").textContent = "Stopped";
  $("viewState").classList.remove("active");
  $("viewCaption").textContent = "Live inspection stopped.";
}
function sendBrowserFrame(socket) {
  if (socket !== state.socket || socket.readyState !== WebSocket.OPEN) return;
  const video = $("browserCamera"), canvas = $("captureCanvas");
  if (!video.videoWidth || !video.videoHeight) { setTimeout(() => sendBrowserFrame(socket), 100); return; }
  const scale = Math.min(1, 960 / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async blob => {
    if (blob && socket === state.socket && socket.readyState === WebSocket.OPEN) socket.send(await blob.arrayBuffer());
  }, "image/jpeg", .78);
}
function handleLiveMessage(event, model, browserSocket) {
  let data;
  try { data = JSON.parse(event.data); } catch { error("Invalid live response from server."); return; }
  if (data.type === "error") { error(data.message); stopLive(); return; }
  if (data.type !== "frame") return;
  setView("live", `data:image/jpeg;base64,${data.image}`, `${model.label} · ${data.width} × ${data.height}`, "Live");
  telemetry(data.inference_ms, data.fps, data.detections.length, model.label);
  $("resultSummary").textContent = `${data.detections.length} detections in current frame · ${data.inference_ms} ms model inference · ${data.fps} delivered FPS.`;
  if (browserSocket && state.socket) setTimeout(() => sendBrowserFrame(state.socket), 70);
}
async function startLive() {
  const model = selectedModel();
  if (!model) { error("Choose an available checkpoint first."); return; }
  clearError(); setDownload(null);
  const kind = $("liveKind").value;
  let path, query = { model_id: model.id, confidence: confidence() };
  try {
    if (kind === "browser") {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Browser camera requires localhost or HTTPS and camera permission.");
      state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 540 } }, audio: false });
      $("browserCamera").srcObject = state.cameraStream;
      await $("browserCamera").play();
      path = "/ws/live/browser";
    } else {
      if (kind === "network" && !state.networkEnabled) throw new Error("Enable ROADEYE_ALLOW_NETWORK_STREAMS=1 when starting the local server.");
      path = "/ws/live/source";
      query = { ...query, kind, source: kind === "usb" ? $("cameraIndex").value : $("streamUrl").value.trim(), max_fps: 15 };
      if (!query.source) throw new Error("Enter a video source.");
    }
    $("startLive").classList.add("hidden"); $("stopLive").classList.remove("hidden");
    $("viewState").textContent = "Connecting";
    const socket = new WebSocket(liveUrl(path, query));
    state.socket = socket;
    socket.onopen = () => { if (kind === "browser") sendBrowserFrame(socket); };
    socket.onmessage = event => handleLiveMessage(event, model, kind === "browser");
    socket.onerror = () => error("Live connection failed. Check server logs, camera access and checkpoint availability.");
    socket.onclose = () => { if (state.socket === socket) stopLive(); };
  } catch (cause) { stopLive(); error(cause.message); }
}
async function scanCameras() {
  $("cameraHint").textContent = "Checking cameras on the server PC…";
  try {
    const response = await fetch("/api/cameras");
    if (!response.ok) throw new Error(await apiError(response));
    const data = await response.json();
    $("cameraHint").textContent = data.cameras.length ? `Available: ${data.cameras.map(c => c.index).join(", ")}.` : "No USB camera detected. Check Windows privacy permissions and the device connection.";
    const select = $("cameraIndex"); select.replaceChildren();
    for (const camera of data.cameras) { const option = document.createElement("option"); option.value = camera.index; option.textContent = camera.label; select.appendChild(option); }
  } catch (cause) { $("cameraHint").textContent = cause.message; }
}
function wireDropzone(inputId, zoneId) {
  const input = $(inputId), zone = $(zoneId);
  zone.addEventListener("dragover", event => { event.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", event => {
    event.preventDefault(); zone.classList.remove("drag-over");
    if (event.dataTransfer.files.length) { input.files = event.dataTransfer.files; input.dispatchEvent(new Event("change")); }
  });
  input.addEventListener("change", () => { $(`${inputId === "imageFile" ? "image" : "video"}Filename`).textContent = input.files[0]?.name || "No file selected"; });
}
async function init() {
  document.querySelectorAll(".mode-tab").forEach(tab => tab.addEventListener("click", () => switchMode(tab.dataset.mode)));
  $("confidence").addEventListener("input", () => $("confidenceValue").textContent = confidence().toFixed(2));
  $("modelSelect").addEventListener("change", updateModelHint);
  $("runImage").addEventListener("click", inspectImage);
  $("runVideo").addEventListener("click", inspectVideo);
  $("startLive").addEventListener("click", startLive);
  $("stopLive").addEventListener("click", stopLive);
  $("scanCameras").addEventListener("click", scanCameras);
  $("liveKind").addEventListener("change", () => {
    $("usbOptions").classList.toggle("hidden", $("liveKind").value !== "usb");
    $("networkOptions").classList.toggle("hidden", $("liveKind").value !== "network");
  });
  wireDropzone("imageFile", "imageDropzone"); wireDropzone("videoFile", "videoDropzone");
  try {
    const [healthResponse, modelsResponse] = await Promise.all([fetch("/api/health"), fetch("/api/models")]);
    if (!healthResponse.ok || !modelsResponse.ok) throw new Error("The local inference server did not respond.");
    const health = await healthResponse.json(), inventory = await modelsResponse.json();
    state.models = inventory.models;
    state.networkEnabled = health.network_streams_enabled;
    $("networkHint").textContent = health.network_streams_enabled ? "Use an RTSP/HTTP URL reachable from the server PC." : "Set ROADEYE_ALLOW_NETWORK_STREAMS=1 to enable network streams on this local server.";
    renderModels();
    setStatus(health.gpu ? `Ready · ${health.gpu}` : "Ready · CPU inference", "ready");
  } catch (cause) { setStatus("Server unavailable", "error"); error(cause.message); }
}
init();
