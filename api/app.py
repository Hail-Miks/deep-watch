from collections import deque
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

from flask import Flask, request, jsonify, render_template, Response, send_file
import numpy as np
import torch
import cv2
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deep_watch import PoseDetector  # noqa: E402
from deep_watch.database import get_incident_logger

# ============================================================================
# PERFORMANCE CONFIGURATION - Tune these for edge devices / lower-end laptops
# ============================================================================
#

# PRESETS: Uncomment ONE preset below, or customize individual settings
#
# ---------- PRESET: HIGH PERFORMANCE (powerful GPU/desktop) ----------
# PROCESS_EVERY_N_FRAMES = 1
# INPUT_SCALE = 1.0
# LSTM_INFERENCE_INTERVAL = 3
# YOLO_IMGSZ = 640
# JPEG_QUALITY = 90
#
# ---------- PRESET: BALANCED (default - moderate hardware) ----------
# PROCESS_EVERY_N_FRAMES = 2
# INPUT_SCALE = 1.0
# LSTM_INFERENCE_INTERVAL = 5
# YOLO_IMGSZ = 640
# JPEG_QUALITY = 80
#
# ---------- PRESET: LOW-END LAPTOP ----------
# PROCESS_EVERY_N_FRAMES = 3
# INPUT_SCALE = 0.75
# LSTM_INFERENCE_INTERVAL = 10
# YOLO_IMGSZ = 480
# JPEG_QUALITY = 70
#
# ---------- PRESET: EDGE DEVICE / RASPBERRY PI ----------
# PROCESS_EVERY_N_FRAMES = 4
# INPUT_SCALE = 0.5
# LSTM_INFERENCE_INTERVAL = 15
# YOLO_IMGSZ = 320
# JPEG_QUALITY = 60
#
# ============================================================================


class PerfConfig:
    """
    Performance settings - EDIT THESE VALUES DIRECTLY to tune for your hardware.

    Lower values = better quality but slower
    Higher values = faster but lower quality/accuracy
    """

    # ==================== EDIT THESE VALUES ====================

    # Frame Processing
    # 1=all frames, 2=every other, 3=every 3rd (higher=faster)
    PROCESS_EVERY_N_FRAMES: int = 2
    # 1.0=full, 0.75=75%, 0.5=half resolution (lower=faster)
    INPUT_SCALE: float = 1.0

    # LSTM Inference Throttling
    # Run LSTM every N processed frames (higher=faster)
    LSTM_INFERENCE_INTERVAL: int = 5
    # Min frames before first inference (lower=faster response)
    LSTM_MIN_BUFFER_FRAMES: int = 15
    # Batch all tracks together (keep True for efficiency)
    LSTM_BATCH_INFERENCE: bool = True

    # Output Quality
    # 50-95, lower=faster/smaller, higher=better quality
    JPEG_QUALITY: int = 80
    # 1.0=full, 0.75=75% output size (lower=faster)
    OUTPUT_SCALE: float = 1.0

    # Memory Management
    MAX_TRACK_AGE_FRAMES: int = 300       # Remove tracks not seen for N frames
    TRACK_CLEANUP_INTERVAL: int = 100     # Run cleanup every N frames

    # Model Settings
    # YOLO input: 640 (accurate), 480, 320 (fast)
    YOLO_IMGSZ: int = 640
    # True=FP16 on CUDA (faster but less accurate)
    HALF_PRECISION: bool = False
    # Minimum confidence to classify as drowning (higher = fewer false positives)
    DROWNING_CONF_THRESHOLD: float = 0.75
    # Require N consecutive drowning predictions before alerting
    DROWNING_SMOOTHING_FRAMES: int = 3

    # ==================== END CONFIGURATION ====================


app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
    static_url_path="/static",
)


def forward_fill_nan(a: np.ndarray) -> np.ndarray:
    b = a.copy()
    for t in range(1, b.shape[0]):
        mask = np.isnan(b[t])
        b[t][mask] = b[t-1][mask]
    for t in range(b.shape[0] - 2, -1, -1):
        mask = np.isnan(b[t])
        b[t][mask] = b[t+1][mask]
    b = np.nan_to_num(b, nan=0.0)
    return b


def normalize_keypoints(kpts: np.ndarray, conf: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    T, K, _ = kpts.shape
    out = kpts.copy().astype(np.float32)
    for t in range(T):
        frame = out[t]
        conf_t = conf[t] if conf is not None and len(
            conf) == T else np.zeros(K)

        def valid_pair(i, j, thr=0.2):
            return conf_t[i] > thr and conf_t[j] > thr and not (np.any(np.isnan(frame[i])) or np.any(np.isnan(frame[j])))

        cx, cy = 0.0, 0.0
        if valid_pair(11, 12):
            cx = 0.5 * (frame[11, 0] + frame[12, 0])
            cy = 0.5 * (frame[11, 1] + frame[12, 1])
        elif valid_pair(5, 6):
            cx = 0.5 * (frame[5, 0] + frame[6, 0])
            cy = 0.5 * (frame[5, 1] + frame[6, 1])
        elif not np.any(np.isnan(bbox[t])):
            x1, y1, x2, y2 = bbox[t]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        else:
            valid = ~np.isnan(frame).any(axis=1)
            if valid.any():
                cx = float(np.mean(frame[valid, 0]))
                cy = float(np.mean(frame[valid, 1]))
            else:
                cx, cy = 0.0, 0.0
        frame[:, 0] -= cx
        frame[:, 1] -= cy
        scale = 1.0
        if valid_pair(5, 6):
            dx = frame[5, 0] - frame[6, 0]
            dy = frame[5, 1] - frame[6, 1]
            scale = float(np.hypot(dx, dy))
        elif valid_pair(11, 12):
            dx = frame[11, 0] - frame[12, 0]
            dy = frame[11, 1] - frame[12, 1]
            scale = float(np.hypot(dx, dy))
        elif not np.any(np.isnan(bbox[t])):
            x1, y1, x2, y2 = bbox[t]
            scale = float(max(1e-6, y2 - y1))
        scale = max(scale, 1e-6)
        frame[:, 0] /= scale
        frame[:, 1] /= scale
        out[t] = frame
    return out


def build_features(kpts: np.ndarray, conf: np.ndarray, bbox: np.ndarray, include_vel: bool = True, include_conf: bool = True) -> np.ndarray:
    kpts_filled = forward_fill_nan(kpts)
    conf_filled = forward_fill_nan(conf[:, :, None])[
        :, :, 0] if conf is not None else None
    bbox_filled = forward_fill_nan(bbox)
    kpts_norm = normalize_keypoints(kpts_filled, conf_filled, bbox_filled)
    pos = kpts_norm.reshape(kpts_norm.shape[0], -1)  # (T, 34)
    parts = [pos]
    if include_vel:
        vel = np.vstack([np.zeros_like(pos[:1]), np.diff(pos, axis=0)])
        parts.append(vel)
    if include_conf and conf_filled is not None:
        parts.append(conf_filled)  # (T, 17) — keypoint detection confidence
    feats = np.concatenate(parts, axis=1)
    return feats.astype(np.float32)


def choose_person(people: List[Dict]) -> Dict:
    if not people:
        return None

    def score(p: Dict) -> Tuple[float, float]:
        confs = [kp["confidence"] for kp in p.get("keypoints", {}).values()]
        avg = float(np.mean(confs)) if confs else 0.0
        bb = p.get("bbox")
        area = (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"]) if bb else 0.0
        return avg, area

    return sorted(people, key=score, reverse=True)[0]


# Model init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POSE = PoseDetector(
    model_path="model/yolov8n-pose.pt",
    conf_threshold=0.15,
    device=DEVICE,
    imgsz=PerfConfig.YOLO_IMGSZ
)

# Log performance configuration on startup
print(f"[PerfConfig] Device: {DEVICE}")
print(f"[PerfConfig] Process every {PerfConfig.PROCESS_EVERY_N_FRAMES} frames")
print(f"[PerfConfig] Input scale: {PerfConfig.INPUT_SCALE}")
print(
    f"[PerfConfig] LSTM inference every {PerfConfig.LSTM_INFERENCE_INTERVAL} processed frames")
print(f"[PerfConfig] YOLO image size: {PerfConfig.YOLO_IMGSZ}")
print(f"[PerfConfig] JPEG quality: {PerfConfig.JPEG_QUALITY}")
print(f"[PerfConfig] Half precision: {PerfConfig.HALF_PRECISION}")

CKPT_PATH = os.environ.get(
    "LSTM_CKPT", os.path.join("runs", "lstm", "best.pt"))
if not os.path.isfile(CKPT_PATH):
    print(
        f"Warning: LSTM checkpoint not found at {CKPT_PATH}. Inference will fail until provided.")


# ============================================================================
# SCENE CLASSIFICATION
# ============================================================================

def classify_scene(video_source, num_frames: int = 30,
                   blue_green_threshold: float = 0.35) -> str:
    """
    basically classifies whether the video feed is underwater or above-water.

    checks the SCENE_TYPE env var first.
    if SCENE_type is unset or "" then we proceed to analyze the first `num_frames` 
    frames for blue-green color dominance using HSV color space.

    returns "underwater" or "abovewater".
    """
    # Manual override just type in here     ↓↓  ("underwater" or "abovewater"), if its blank then it is unset by default
    override = os.environ.get("SCENE_TYPE", "").strip().lower()
    if override in ("underwater", "abovewater"):
        return override

    #Auto-detecting scene from first {num_frames} frames" (oNly happens if the SCENE_TYPE env var is not set to "underwater" or "abovewater")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return "underwater" # if there is not source camera input, we default to underwater

    # calculate each frame's blue-green ratio to classify whether underwater or above water
    blue_green_ratios = []
    frames_read = 0
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, _v = cv2.split(hsv)
        # blue-green hue range (80-140)
        # saturation (>40) to ignore gray/white areas
        # can edit thes thresholds
        mask = (h >= 80) & (h <= 140) & (s > 40)
        ratio = float(np.count_nonzero(mask)) / max(1, mask.size)
        blue_green_ratios.append(ratio)

    cap.release()

    # just incase for some reason no frames are read, we default to underwater to be safe
    if frames_read == 0:
        return "underwater"

    avg_ratio = float(np.mean(blue_green_ratios))
    scene = "underwater" if avg_ratio >= blue_green_threshold else "abovewater"
    print(f"[SceneMode] Blue-green ratio: {avg_ratio:.3f} (threshold: {blue_green_threshold}) -> {scene}")
    return scene


# Live statistics (thread-safe)
_stats_lock = threading.Lock()
_live_stats = {
    "fps": 0.0,
    "frame_count": 0,
    "processed_frames": 0,
    "skipped_frames": 0,
    "people_tracked": 0,
    "active_tracks": 0,
    "drowning_alerts": 0,
    "drowning_track_ids": [],
    "start_time": None,
    "uptime_seconds": 0,
    "model_loaded": False,
    "camera_connected": False,
    "lstm_inferences": 0,
    "avg_inference_ms": 0.0,
    "scene_mode": "unknown",
}


def update_stats(**kwargs):
    with _stats_lock:
        _live_stats.update(kwargs)
        if _live_stats["start_time"] is not None:
            _live_stats["uptime_seconds"] = time.time() - \
                _live_stats["start_time"]


def get_stats():
    with _stats_lock:
        stats = _live_stats.copy()
        if stats["start_time"] is not None:
            stats["uptime_seconds"] = time.time() - stats["start_time"]
        return stats


# Live statistics (thread-safe)
_stats_lock = threading.Lock()
_live_stats = {
    "fps": 0.0,
    "frame_count": 0,
    "processed_frames": 0,
    "skipped_frames": 0,
    "people_tracked": 0,
    "active_tracks": 0,
    "drowning_alerts": 0,
    "drowning_track_ids": [],
    "start_time": None,
    "uptime_seconds": 0,
    "model_loaded": False,
    "camera_connected": False,
    "lstm_inferences": 0,
    "avg_inference_ms": 0.0,
    "scene_mode": "unknown",
}

def update_stats(**kwargs):
    with _stats_lock:
        _live_stats.update(kwargs)
        if _live_stats["start_time"] is not None:
            _live_stats["uptime_seconds"] = time.time() - _live_stats["start_time"]

def get_stats():
    with _stats_lock:
        stats = _live_stats.copy()
        if stats["start_time"] is not None:
            stats["uptime_seconds"] = time.time() - stats["start_time"]
        return stats

_ckpt = None
_model = None
_class_to_idx = None
_window = None
_input_size = None


def load_lstm():
    global _ckpt, _model, _class_to_idx, _window, _input_size
    if _model is not None and _class_to_idx is not None:
        return _model, _class_to_idx, _window
    _ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    from training.train_lstm import LSTMClassifier
    _input_size = int(_ckpt.get("input_size"))
    hidden = int(_ckpt.get("hidden", 128))
    layers = int(_ckpt.get("layers", 1))
    _class_to_idx = _ckpt.get("class_to_idx")
    _window = int(_ckpt.get("window", 60))
    _model = LSTMClassifier(input_size=_input_size, hidden_size=hidden,
                            num_layers=layers, num_classes=len(_class_to_idx)).to(DEVICE)
    _model.load_state_dict(_ckpt["model_state"])
    _model.eval()
    return _model, _class_to_idx, _window


# Video processing

def extract_sequence_from_video(video_path: str) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    T = len(frames)
    K = len(POSE.keypoint_names)
    kpts = np.full((T, K, 2), np.nan, dtype=np.float32)
    conf = np.zeros((T, K), dtype=np.float32)
    bbox = np.full((T, 4), np.nan, dtype=np.float32)

    for t, frame in enumerate(frames):
        results, _ = POSE.detect_image(frame, visualize=False)
        if not results:
            continue
        result = results[0]
        people = POSE.get_keypoints(result)
        person = choose_person(people)
        if person is None:
            continue
        for i, name in enumerate(POSE.keypoint_names):
            kp = person["keypoints"].get(name)
            if kp is None:
                continue
            kpts[t, i, 0] = kp["x"]
            kpts[t, i, 1] = kp["y"]
            conf[t, i] = kp["confidence"]
        if person.get("bbox"):
            bb = person["bbox"]
            bbox[t] = [bb["x1"], bb["y1"], bb["x2"], bb["y2"]]

    return fps, kpts, conf, bbox


# Routes

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/stats")
def stats():
    """Return live statistics about the video feed and detections."""
    s = get_stats()
    return jsonify({
        "fps": round(s["fps"], 1),
        "frame_count": s["frame_count"],
        "processed_frames": s.get("processed_frames", 0),
        "skipped_frames": s.get("skipped_frames", 0),
        "people_tracked": s["people_tracked"],
        "active_tracks": s["active_tracks"],
        "drowning_alerts": s["drowning_alerts"],
        "drowning_track_ids": s["drowning_track_ids"],
        "uptime_seconds": int(s["uptime_seconds"]),
        "model_loaded": s["model_loaded"],
        "camera_connected": s["camera_connected"],
        "lstm_inferences": s.get("lstm_inferences", 0),
        "avg_inference_ms": round(s.get("avg_inference_ms", 0), 1),
        "scene_mode": s.get("scene_mode", "unknown"),
    })


@app.get("/config")
def get_config():
    """Return current performance configuration."""
    return jsonify({
        "process_every_n_frames": PerfConfig.PROCESS_EVERY_N_FRAMES,
        "input_scale": PerfConfig.INPUT_SCALE,
        "lstm_inference_interval": PerfConfig.LSTM_INFERENCE_INTERVAL,
        "lstm_min_buffer_frames": PerfConfig.LSTM_MIN_BUFFER_FRAMES,
        "jpeg_quality": PerfConfig.JPEG_QUALITY,
        "yolo_imgsz": PerfConfig.YOLO_IMGSZ,
        "half_precision": PerfConfig.HALF_PRECISION,
        "device": DEVICE,
        "scene_mode": _live_stats.get("scene_mode", "unknown"),
    })


@app.post("/predict_video")
def predict_video():
    try:
        model, class_to_idx, window = load_lstm()
    except Exception as e:
        return jsonify({"error": f"Failed to load LSTM checkpoint: {str(e)}"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part 'file' in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, file.filename)
        file.save(path)

        try:
            fps, kpts, conf, bbox = extract_sequence_from_video(path)
        except Exception as e:
            return jsonify({"error": f"Video processing failed: {str(e)}"}), 400

    feats = build_features(kpts, conf, bbox, include_vel=True)  # (T, F)
    F = feats.shape[1]
    if F != model.fc.in_features:
        pass

    # Build windows for inference
    windows = []
    starts = list(range(0, max(1, feats.shape[0] - window + 1), window))
    if not starts:
        starts = [0]
    for s in starts:
        w = feats[s:s+window]
        if w.shape[0] < window:
            pad = np.zeros(
                (window - w.shape[0], feats.shape[1]), dtype=w.dtype)
            w = np.vstack([w, pad])
        windows.append(w)

    x = torch.from_numpy(np.stack(windows, axis=0)).to(DEVICE)  # (B, T, F)
    with torch.no_grad():
        logits = model(x)  # (B, C)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    mean_probs = probs.mean(axis=0)

    # Build class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_idx = int(np.argmax(mean_probs))
    pred_class = idx_to_class.get(pred_idx, str(pred_idx))
    per_class = {idx_to_class[i]: float(p) for i, p in enumerate(mean_probs)}

    return jsonify({
        "predicted_class": pred_class,
        "probabilities": per_class,
        "num_windows": int(len(windows)),
        "window_size": int(window),
        "fps": float(fps),
    })


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/incidents_view")
def incidents_view():
    return render_template("incidents.html")


@app.get("/incidents")
def incidents():
    try:
        limit = int(request.args.get("limit", 200))
    except (TypeError, ValueError):
        limit = 200

    incident_logger = get_incident_logger()
    return jsonify(incident_logger.get_all_incidents(limit=limit))


def generate_stream(video_source: int | str):
    """Optimized video stream generator with performance controls."""
    # classify scene on startup (underwater vs abovewater)
    scene_mode = classify_scene(video_source)
    update_stats(scene_mode=scene_mode)

    model_loaded = False
    if scene_mode == "underwater":
        try:
            model, class_to_idx, window = load_lstm()
            model_loaded = True
            # Enable half precision if configured and on CUDA
            if PerfConfig.HALF_PRECISION and DEVICE == "cuda":
                model = model.half()
        except Exception:
            model, class_to_idx, window = None, None, 60
    else:
        # Above-water mode: skip LSTM entirely
        model, class_to_idx, window = None, None, 60
        print("[SceneMode] Above-water — LSTM classification disabled")

    update_stats(model_loaded=model_loaded, start_time=time.time())

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        update_stats(camera_connected=False)

        def gen_error():
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(img, "No camera input detected", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            ret, buf = cv2.imencode('.jpg', img)
            if ret:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        yield from gen_error()
        return
    
    update_stats(camera_connected=True)

    update_stats(camera_connected=True)

    K = len(POSE.keypoint_names)

    # Per-track buffers for multi-person tracking with ByteTrack
    track_kbufs = {}  # track_id -> deque of keypoints
    track_cbufs = {}  # track_id -> deque of confidences
    track_bbufs = {}  # track_id -> deque of bboxes
    track_predictions = {}  # track_id -> (pred_class, confidence, color)
    track_last_seen = {}  # track_id -> frame number when last seen
    track_alert_bufs = {}  # track_id -> deque of recent drowning flags
    drowning_incident_times = {}  # track_id -> timestamp of last drowning detection (persists through occlusion)

    idx_to_class = None
    if class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Performance tracking
    frame_count = 0
    processed_frame_count = 0
    skipped_frames = 0
    start_time = time.time()
    fps_update_interval = 10
    lstm_inference_count = 0
    total_inference_time_ms = 0.0

    # JPEG encoding params
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, PerfConfig.JPEG_QUALITY]

    # Cache for last visualization (used when skipping frames)
    last_vis = None
    last_vis_bytes = None
    drowning_ids = []
    active_drowning_ids = set()  # Persists across frames for occlusion handling
    people = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1

            # Frame skipping - only process every N frames
            should_process = (frame_count %
                              PerfConfig.PROCESS_EVERY_N_FRAMES == 0)

            if not should_process:
                skipped_frames += 1
                # Return cached frame if available
                if last_vis_bytes is not None:
                    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + last_vis_bytes + b"\r\n")
                continue

            processed_frame_count += 1

            # Downscale input if configured
            process_frame = frame
            original_h, original_w = frame.shape[:2]
            if PerfConfig.INPUT_SCALE < 1.0:
                new_w = int(original_w * PerfConfig.INPUT_SCALE)
                new_h = int(original_h * PerfConfig.INPUT_SCALE)
                process_frame = cv2.resize(
                    frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Use YOLOv8's built-in tracking with custom ByteTrack config
            results, _ = POSE.track(process_frame, tracker_type=os.path.join(
                PROJECT_ROOT, "tracker", "bytetrack_custom.yaml"))

            # Use original frame for visualization
            vis = frame.copy()

            if results:
                result = results[0]

                # Scale keypoints back to original resolution if downscaled
                scale_factor = 1.0 / PerfConfig.INPUT_SCALE if PerfConfig.INPUT_SCALE < 1.0 else 1.0

                try:
                    vis = result.plot(labels=False, conf=False)
                    # Resize visualization back to original if needed
                    if PerfConfig.INPUT_SCALE < 1.0:
                        vis = cv2.resize(
                            vis, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    vis = frame.copy()

                # Extract tracked people with their IDs
                people = POSE.get_keypoints(result)

                # Update per-track buffers and predictions
                if hasattr(result, 'boxes') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(people)))

                current_track_ids = set()
                for person_idx, (person, track_id) in enumerate(zip(people, track_ids)):
                    track_id = int(track_id)
                    current_track_ids.add(track_id)

                    # Initialize buffers if new track
                    if track_id not in track_kbufs:
                        track_kbufs[track_id] = deque(maxlen=window)
                        track_cbufs[track_id] = deque(maxlen=window)
                        track_bbufs[track_id] = deque(maxlen=window)
                        track_predictions[track_id] = (
                            "Unknown", 0.5, (0, 255, 255))
                        track_alert_bufs[track_id] = deque(
                            maxlen=PerfConfig.DROWNING_SMOOTHING_FRAMES)

                    track_last_seen[track_id] = frame_count

                    # Extract and store keypoints (scale back if needed)
                    b = person.get("bbox", {})
                    bb = [
                        b.get("x1", 0) * scale_factor,
                        b.get("y1", 0) * scale_factor,
                        b.get("x2", 100) * scale_factor,
                        b.get("y2", 100) * scale_factor
                    ]

                    k = np.full((K, 2), np.nan, dtype=np.float32)
                    c = np.zeros((K,), dtype=np.float32)
                    for i, name in enumerate(POSE.keypoint_names):
                        kp = person["keypoints"].get(name)
                        if kp is not None:
                            k[i, 0] = kp["x"] * scale_factor
                            k[i, 1] = kp["y"] * scale_factor
                            c[i] = kp["confidence"]

                    track_kbufs[track_id].append(k)
                    track_cbufs[track_id].append(c)
                    track_bbufs[track_id].append(bb)

                # LSTM inference - throttled and batched (underwater mode only)
                drowning_ids = []
                should_run_lstm = (
                    scene_mode == "underwater" and
                    model is not None and
                    processed_frame_count % PerfConfig.LSTM_INFERENCE_INTERVAL == 0
                )

                if should_run_lstm:
                    inference_start = time.time()

                    # Collect tracks eligible for inference
                    eligible_tracks = []
                    for track_id in current_track_ids:
                        track_id = int(track_id)
                        if len(track_kbufs.get(track_id, [])) >= PerfConfig.LSTM_MIN_BUFFER_FRAMES:
                            eligible_tracks.append(track_id)

                    if eligible_tracks:
                        if PerfConfig.LSTM_BATCH_INFERENCE and len(eligible_tracks) > 1:
                            # Batched inference - process all tracks at once
                            batch_features = []
                            for track_id in eligible_tracks:
                                kbuf = track_kbufs[track_id]
                                cbuf = track_cbufs[track_id]
                                bbuf = track_bbufs[track_id]

                                karr = np.stack(list(kbuf), axis=0)
                                carr = np.stack(list(cbuf), axis=0)
                                barr = np.stack(list(bbuf), axis=0)
                                feats = build_features(
                                    karr, carr, barr, include_vel=True)

                                if feats.shape[0] < window:
                                    pad = np.zeros(
                                        (window - feats.shape[0], feats.shape[1]), dtype=feats.dtype)
                                    wfeats = np.vstack([feats, pad])
                                else:
                                    wfeats = feats[-window:]
                                batch_features.append(wfeats)

                            # Single batched forward pass
                            x = torch.from_numpy(
                                np.stack(batch_features, axis=0)).to(DEVICE)
                            if PerfConfig.HALF_PRECISION and DEVICE == "cuda":
                                x = x.half()

                            with torch.no_grad():
                                logits = model(x)
                                probs = torch.softmax(
                                    logits, dim=1).cpu().numpy()

                            # Process results
                            for i, track_id in enumerate(eligible_tracks):
                                if idx_to_class:
                                    pred_idx = int(np.argmax(probs[i]))
                                    pred_class = idx_to_class.get(
                                        pred_idx, str(pred_idx))
                                    pred_conf = float(probs[i, pred_idx])
                                    raw_drowning = pred_class.lower().startswith('drowning') and pred_conf > PerfConfig.DROWNING_CONF_THRESHOLD
                                    if PerfConfig.DROWNING_SMOOTHING_FRAMES <= 1:
                                        is_drowning = raw_drowning
                                    else:
                                        buf = track_alert_bufs.setdefault(
                                            track_id, deque(maxlen=PerfConfig.DROWNING_SMOOTHING_FRAMES))
                                        buf.append(raw_drowning)
                                        is_drowning = len(buf) == buf.maxlen and all(buf)
                                    color = (0, 0, 255) if is_drowning else (
                                        0, 200, 0)
                                    track_predictions[track_id] = (
                                        pred_class, pred_conf, color)
                                    if is_drowning:
                                        drowning_ids.append(track_id)
                                        drowning_incident_times[track_id] = time.time()
                        else:
                            # Single track inference (or batch disabled)
                            for track_id in eligible_tracks:
                                kbuf = track_kbufs[track_id]
                                cbuf = track_cbufs[track_id]
                                bbuf = track_bbufs[track_id]

                                karr = np.stack(list(kbuf), axis=0)
                                carr = np.stack(list(cbuf), axis=0)
                                barr = np.stack(list(bbuf), axis=0)
                                feats = build_features(
                                    karr, carr, barr, include_vel=True)

                                if feats.shape[0] < window:
                                    pad = np.zeros(
                                        (window - feats.shape[0], feats.shape[1]), dtype=feats.dtype)
                                    wfeats = np.vstack([feats, pad])
                                else:
                                    wfeats = feats[-window:]

                                x = torch.from_numpy(
                                    wfeats[None, ...]).to(DEVICE)
                                if PerfConfig.HALF_PRECISION and DEVICE == "cuda":
                                    x = x.half()

                                with torch.no_grad():
                                    logits = model(x)
                                    probs = torch.softmax(
                                        logits, dim=1).cpu().numpy()[0]

                                if idx_to_class:
                                    pred_idx = int(np.argmax(probs))
                                    pred_class = idx_to_class.get(
                                        pred_idx, str(pred_idx))
                                    pred_conf = float(probs[pred_idx])
                                    raw_drowning = pred_class.lower().startswith('drowning') and pred_conf > PerfConfig.DROWNING_CONF_THRESHOLD
                                    if PerfConfig.DROWNING_SMOOTHING_FRAMES <= 1:
                                        is_drowning = raw_drowning
                                    else:
                                        buf = track_alert_bufs.setdefault(
                                            track_id, deque(maxlen=PerfConfig.DROWNING_SMOOTHING_FRAMES))
                                        buf.append(raw_drowning)
                                        is_drowning = len(buf) == buf.maxlen and all(buf)
                                    color = (0, 0, 255) if is_drowning else (
                                        0, 200, 0)
                                    track_predictions[track_id] = (
                                        pred_class, pred_conf, color)
                                    if is_drowning:
                                        drowning_ids.append(track_id)
                                        drowning_incident_times[track_id] = time.time()

                        lstm_inference_count += 1
                        inference_time = (time.time() - inference_start) * 1000
                        total_inference_time_ms += inference_time
                
                # Clean up dead tracks (not seen for > 30 frames)
                # This prevents stale drowning predictions from triggering alerts after occlusion
                dead_track_ids = [
                    tid for tid in track_last_seen
                    if (frame_count - track_last_seen[tid]) > 30
                ]
                for tid in dead_track_ids:
                    # Reset drowning state so ghost alerts don't fire
                    track_predictions[tid] = ("Unknown", 0.5, (0, 255, 255))
                    track_alert_bufs[tid].clear()
                    # Clean up the oldest buffers if memory grows too large
                    if len(track_kbufs) > 50:
                        del track_kbufs[tid]
                        del track_cbufs[tid]
                        del track_bbufs[tid]
                        del track_last_seen[tid]
                
                # Log drowning incidents with cooldown (underwater mode only)
                # Alert persists through brief occlusions for up to 5 seconds
                now = time.time()
                active_drowning_ids = set(drowning_ids)  # Currently visible drowning
                # Add recently drowning tracks (< 5 sec ago) even if occluded
                recent_drowning_ids = [
                    tid for tid, t in drowning_incident_times.items()
                    if (now - t) < 5.0
                ]
                active_drowning_ids.update(recent_drowning_ids)
                
                # Clean up old incidents
                drowning_incident_times = {
                    tid: t for tid, t in drowning_incident_times.items()
                    if (now - t) < 5.0
                }
                
                if active_drowning_ids and scene_mode == "underwater":
                    if not hasattr(generate_stream, 'last_log_time'):
                        generate_stream.last_log_time = 0.0
                    if now - generate_stream.last_log_time >= 5.0:
                        incident_logger = get_incident_logger()
                        incident_logger.log_incident(
                            track_ids=list(active_drowning_ids),
                            description="Drowning Detected",
                        )
                        generate_stream.last_log_time = now
                        
                # Draw track IDs and predictions on visualization
                h, w = vis.shape[:2]
                if hasattr(result, 'boxes') and result.boxes.id is not None:
                    for box, track_id in zip(result.boxes.xyxy, track_ids):
                        track_id = int(track_id)
                        # Scale box coordinates if input was downscaled
                        x1, y1, x2, y2 = [int(coord * scale_factor)
                                          for coord in box[:4]]
                        pred_class, pred_conf, color = track_predictions.get(
                            track_id, ("Unknown", 0.5, (0, 255, 255)))

                        # Above-water mode: show track ID only, no classification OR LSTMM
                        if scene_mode == "abovewater":
                            label = f"ID:{track_id}"
                            color = (0, 255, 255)  # Neutral yellow
                        else:
                            label = f"ID:{track_id} {pred_class} ({pred_conf:.2f})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        thickness = 2
                        text_size = cv2.getTextSize(
                            label, font, font_scale, thickness)[0]

                        # Position at top-left of bbox
                        label_x = x1
                        label_y = max(y1 - 5, text_size[1] + 5)

                        # Background rectangle
                        cv2.rectangle(vis,
                                      (label_x - 2, label_y -
                                       text_size[1] - 5),
                                      (label_x +
                                       text_size[0] + 5, label_y + 3),
                                      color, -1)

                        # Text color
                        cv2.putText(vis, label, (label_x, label_y),
                                    font, font_scale, (255, 255, 255), thickness)

                # Memory cleanup - remove old tracks
                if frame_count % PerfConfig.TRACK_CLEANUP_INTERVAL == 0:
                    stale_tracks = [
                        tid for tid, last_seen in track_last_seen.items()
                        if frame_count - last_seen > PerfConfig.MAX_TRACK_AGE_FRAMES
                    ]
                    for tid in stale_tracks:
                        track_kbufs.pop(tid, None)
                        track_cbufs.pop(tid, None)
                        track_bbufs.pop(tid, None)
                        track_predictions.pop(tid, None)
                        track_last_seen.pop(tid, None)
                        track_alert_bufs.pop(tid, None)
                        drowning_incident_times.pop(tid, None)

            # Downscale output if configured
            if PerfConfig.OUTPUT_SCALE < 1.0:
                out_w = int(vis.shape[1] * PerfConfig.OUTPUT_SCALE)
                out_h = int(vis.shape[0] * PerfConfig.OUTPUT_SCALE)
                vis = cv2.resize(vis, (out_w, out_h),
                                 interpolation=cv2.INTER_LINEAR)

            # Update statistics periodically
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0.0
                avg_inference_ms = total_inference_time_ms / \
                    max(1, lstm_inference_count)
                update_stats(
                    fps=current_fps,
                    frame_count=frame_count,
                    processed_frames=processed_frame_count,
                    skipped_frames=skipped_frames,
                    people_tracked=len(track_kbufs),
                    active_tracks=len(people) if results else 0,
                    drowning_alerts=len(active_drowning_ids),
                    drowning_track_ids=list(active_drowning_ids),
                    lstm_inferences=lstm_inference_count,
                    avg_inference_ms=avg_inference_ms,
                )

            # Encode frame with quality setting
            ret, buf = cv2.imencode('.jpg', vis, encode_params)
            if not ret:
                continue

            # Cache for skipped frames
            last_vis = vis
            last_vis_bytes = buf.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + last_vis_bytes + b"\r\n")

    finally:
        cap.release()


@app.get("/video_feed")
def video_feed():
    source = os.environ.get("VIDEO_SOURCE", "dataset/drowning_sample2.avi")
    # source = 0
    try:
        source_val = int(source)
    except ValueError:
        source_val = source
    return Response(generate_stream(source_val), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
