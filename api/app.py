import os
import sys
import tempfile
from typing import Dict, List, Tuple

from flask import Flask, request, jsonify, render_template, Response
import numpy as np
import torch
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deep_watch import PoseDetector  # noqa: E402


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
        conf_t = conf[t] if conf is not None and len(conf) == T else np.zeros(K)

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


def build_features(kpts: np.ndarray, conf: np.ndarray, bbox: np.ndarray, include_vel: bool = True) -> np.ndarray:
    kpts_filled = forward_fill_nan(kpts)
    conf_filled = forward_fill_nan(conf[:, :, None])[:, :, 0] if conf is not None else None
    bbox_filled = forward_fill_nan(bbox)
    kpts_norm = normalize_keypoints(kpts_filled, conf_filled, bbox_filled)
    pos = kpts_norm.reshape(kpts_norm.shape[0], -1)  # (T, 34)
    if include_vel:
        vel = np.vstack([np.zeros_like(pos[:1]), np.diff(pos, axis=0)])
        feats = np.concatenate([pos, vel], axis=1)
    else:
        feats = pos
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
POSE = PoseDetector(model_path="model/yolov8n-pose.pt", conf_threshold=0.25, device=DEVICE)

CKPT_PATH = os.environ.get("LSTM_CKPT", os.path.join("runs", "lstm", "best.pt"))
if not os.path.isfile(CKPT_PATH):
    print(f"Warning: LSTM checkpoint not found at {CKPT_PATH}. Inference will fail until provided.")

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
    _model = LSTMClassifier(input_size=_input_size, hidden_size=hidden, num_layers=layers, num_classes=len(_class_to_idx)).to(DEVICE)
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
            pad = np.zeros((window - w.shape[0], feats.shape[1]), dtype=w.dtype)
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


from collections import deque


@app.get("/")
def index():
    return render_template("index.html")


def generate_stream(video_source: int | str):
    try:
        model, class_to_idx, window = load_lstm()
    except Exception:
        model, class_to_idx, window = None, None, 60

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        def gen_error():
            img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Camera open failed", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            ret, buf = cv2.imencode('.jpg', img)
            if ret:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        yield from gen_error()
        return

    K = len(POSE.keypoint_names)
    
    # Per-track buffers for multi-person tracking with ByteTrack
    track_kbufs = {}  # track_id -> deque of keypoints
    track_cbufs = {}  # track_id -> deque of confidences
    track_bbufs = {}  # track_id -> deque of bboxes
    track_predictions = {}  # track_id -> (pred_class, confidence, color)

    idx_to_class = None
    if class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Use YOLOv8's built-in tracking
            results, _ = POSE.track(frame)
            vis = frame.copy()

            if results:
                result = results[0]
                try:
                    vis = result.plot(labels=False, conf=False)
                except Exception:
                    vis = frame.copy()
                
                # Extract tracked people with their IDs
                people = POSE.get_keypoints(result)
                
                # Update per-track buffers and predictions
                if hasattr(result, 'boxes') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(people)))
                
                for person_idx, (person, track_id) in enumerate(zip(people, track_ids)):
                    track_id = int(track_id)
                    
                    # Initialize buffers if new track
                    if track_id not in track_kbufs:
                        track_kbufs[track_id] = deque(maxlen=window)
                        track_cbufs[track_id] = deque(maxlen=window)
                        track_bbufs[track_id] = deque(maxlen=window)
                        track_predictions[track_id] = ("Unknown", 0.5, (0, 255, 255))
                    
                    # Extract and store keypoints
                    b = person.get("bbox", {})
                    bb = [b.get("x1", 0), b.get("y1", 0), b.get("x2", 100), b.get("y2", 100)]
                    
                    k = np.full((K, 2), np.nan, dtype=np.float32)
                    c = np.zeros((K,), dtype=np.float32)
                    for i, name in enumerate(POSE.keypoint_names):
                        kp = person["keypoints"].get(name)
                        if kp is not None:
                            k[i, 0] = kp["x"]
                            k[i, 1] = kp["y"]
                            c[i] = kp["confidence"]
                    
                    track_kbufs[track_id].append(k)
                    track_cbufs[track_id].append(c)
                    track_bbufs[track_id].append(bb)
                
                # Perform LSTM inference for each active track
                if model is not None:
                    for track_id in set(track_ids if isinstance(track_ids, list) else track_ids.tolist()):
                        track_id = int(track_id)
                        kbuf = track_kbufs[track_id]
                        cbuf = track_cbufs[track_id]
                        bbuf = track_bbufs[track_id]
                        
                        if len(kbuf) >= 1:
                            karr = np.stack(list(kbuf), axis=0)
                            carr = np.stack(list(cbuf), axis=0)
                            barr = np.stack(list(bbuf), axis=0)
                            feats = build_features(karr, carr, barr, include_vel=True)
                            if feats.shape[0] < window:
                                pad = np.zeros((window - feats.shape[0], feats.shape[1]), dtype=feats.dtype)
                                wfeats = np.vstack([feats, pad])
                            else:
                                wfeats = feats[-window:]
                            
                            x = torch.from_numpy(wfeats[None, ...]).to(DEVICE)
                            with torch.no_grad():
                                logits = model(x)
                                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            
                            if idx_to_class:
                                pred_idx = int(np.argmax(probs))
                                pred_class = idx_to_class.get(pred_idx, str(pred_idx))
                                pred_conf = float(probs[pred_idx])
                                is_drowning = pred_class.lower().startswith('drowning') and pred_conf > 0.5
                                color = (0, 0, 255) if is_drowning else (0, 200, 0)
                                track_predictions[track_id] = (pred_class, pred_conf, color)
                
                # Draw track IDs and predictions on top of Ultralytics visualization
                h, w = vis.shape[:2]
                if hasattr(result, 'boxes') and result.boxes.id is not None:
                    for box, track_id in zip(result.boxes.xyxy, track_ids):
                        track_id = int(track_id)
                        x1, y1, x2, y2 = map(int, box[:4])
                        pred_class, pred_conf, color = track_predictions.get(track_id, ("Unknown", 0.5, (0, 255, 255)))
                        
                        label = f"ID:{track_id} {pred_class} ({pred_conf:.2f})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        thickness = 2
                        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                        
                        # Position at top-left of bbox
                        label_x = x1
                        label_y = max(y1 - 5, text_size[1] + 5)
                        
                        # Background rectangle
                        cv2.rectangle(vis, 
                                    (label_x - 2, label_y - text_size[1] - 5),
                                    (label_x + text_size[0] + 5, label_y + 3),
                                    color, -1)
                        
                        # Text color
                        cv2.putText(vis, label, (label_x, label_y), font, font_scale, (255, 255, 255), thickness)

            ret, buf = cv2.imencode('.jpg', vis)
            if not ret:
                continue
            frame_bytes = buf.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    finally:
        cap.release()


@app.get("/video_feed")
def video_feed():
    source = os.environ.get("VIDEO_SOURCE", "dataset/drowning_sample.avi")
    try:
        source_val = int(source)
    except ValueError:
        source_val = source
    return Response(generate_stream(source_val), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
