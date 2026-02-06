import os
import glob
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


def load_npz_sequences(data_root: str) -> List[Dict]:
    """Load all .npz under data_root/<class>/*.npz
    Returns list of dicts with keys: path, label, kpts, conf, bbox, fps
    """
    items: List[Dict] = []
    classes = []
    for cls in sorted(os.listdir(data_root)):
        cdir = os.path.join(data_root, cls)
        if not os.path.isdir(cdir):
            continue
        classes.append(cls)
        for path in sorted(glob.glob(os.path.join(cdir, "*.npz"))):
            try:
                data = np.load(path)
                kpts = data["kpts"]  # (T, 17, 2)
                conf = data["conf"]  # (T, 17)
                bbox = data["bbox"]  # (T, 4)
                fps = float(data.get("fps", 0.0))
                items.append({
                    "path": path,
                    "label": cls,
                    "kpts": kpts,
                    "conf": conf,
                    "bbox": bbox,
                    "fps": fps,
                })
            except Exception as e:
                print(f"Skip {path}: {e}")
    print(f"Loaded {len(items)} sequences from {len(classes)} classes: {classes}")
    return items


def forward_fill_nan(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along time axis 0. Then back-fill remaining with zeros."""
    b = a.copy()
    # forward fill
    for t in range(1, b.shape[0]):
        mask = np.isnan(b[t])
        b[t][mask] = b[t-1][mask]
    # back fill
    for t in range(b.shape[0] - 2, -1, -1):
        mask = np.isnan(b[t])
        b[t][mask] = b[t+1][mask]
    # replace any remaining NaNs with 0
    b = np.nan_to_num(b, nan=0.0)
    return b


def normalize_keypoints(kpts: np.ndarray, conf: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Normalize keypoints per frame to be roughly translation/scale invariant.
    - Center: average of hips (11,12) if confident; else shoulders (5,6); else bbox center; else mean of all kpts.
    - Scale: shoulder distance if available; else hip distance; else bbox height; else 1.0.
    Returns array of same shape as kpts.
    """
    T, K, _ = kpts.shape
    out = kpts.copy().astype(np.float32)
    for t in range(T):
        frame = out[t]
        conf_t = conf[t] if conf is not None and len(conf) == T else np.zeros(K)
        # pick center
        def valid_pair(i, j, thr=0.2):
            return conf_t[i] > thr and conf_t[j] > thr and not (np.any(np.isnan(frame[i])) or np.any(np.isnan(frame[j])))
        cx, cy = 0.0, 0.0
        if valid_pair(11, 12):  # hips
            cx = 0.5 * (frame[11, 0] + frame[12, 0])
            cy = 0.5 * (frame[11, 1] + frame[12, 1])
        elif valid_pair(5, 6):  # shoulders
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
        # subtract center
        frame[:, 0] -= cx
        frame[:, 1] -= cy
        # scale
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
    """Create per-frame feature vectors.
    - Normalize positions
    - Optionally append velocities (first difference)
    Returns array of shape (T, F)
    """
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


class WindowedKeypointDataset(Dataset):
    def __init__(self, items: List[Dict], class_to_idx: Dict[str, int], window: int = 60, stride: int = 30, include_vel: bool = True):
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.num_features = None
        for it in items:
            kpts = it["kpts"]
            conf = it["conf"]
            bbox = it["bbox"]
            feats = build_features(kpts, conf, bbox, include_vel=include_vel)
            T = feats.shape[0]
            if self.num_features is None:
                self.num_features = feats.shape[1]
            y = class_to_idx[it["label"]]
            # windowing
            if T < window:
                # pad last window with zeros
                pad = np.zeros((window - T, feats.shape[1]), dtype=feats.dtype)
                w = np.vstack([feats, pad])
                self.samples.append((w, y))
            else:
                for start in range(0, T - window + 1, stride):
                    w = feats[start:start + window]
                    self.samples.append((w, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # return as torch tensors (T, F)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        # Dropout is ignored by PyTorch when num_layers == 1; we keep parameter for flexibility
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == y).sum().item())
        total += x.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


def eval_model(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == y).sum().item())
            total += x.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


def build_clipwise_split(items: List[Dict], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split by clip (items), stratified approximately by class.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[Dict]] = {}
    for it in items:
        by_class.setdefault(it["label"], []).append(it)
    train_items, val_items = [], []
    for cls, arr in by_class.items():
        n = len(arr)
        idx = np.arange(n)
        rng.shuffle(idx)
        split = max(1, int(np.floor((1.0 - val_ratio) * n)))
        train_idx, val_idx = idx[:split], idx[split:]
        train_items.extend([arr[i] for i in train_idx])
        val_items.extend([arr[i] for i in val_idx])
    return train_items, val_items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.path.join("dataset", "keypoints"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=90)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout for LSTM (ignored when layers=1)")
    p.add_argument("--class-weights", default="auto", choices=["none", "auto"], help="Use class weights in loss (auto = inverse frequency)")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu"
    )
    print(f"Using device: {device}")

    items = load_npz_sequences(args.data_root)
    class_names = sorted({it["label"] for it in items})
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Classes: {class_to_idx}")

    train_items, val_items = build_clipwise_split(items, val_ratio=args.val_ratio)
    train_ds = WindowedKeypointDataset(train_items, class_to_idx, window=args.window, stride=args.stride, include_vel=True)
    val_ds = WindowedKeypointDataset(val_items, class_to_idx, window=args.window, stride=args.stride, include_vel=True)

    input_size = train_ds.num_features
    print(f"Feature size: {input_size}, Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LSTMClassifier(input_size=input_size, hidden_size=args.hidden, num_layers=args.layers, num_classes=len(class_names), dropout=args.dropout).to(device)

    # Optional class weighting to address imbalance
    if args.class_weights == "auto":
        counts = {c: 0 for c in class_names}
        for it in items:
            counts[it["label"]] += 1
        weights = []
        total = sum(counts.values())
        for c in class_names:
            cnt = max(1, counts[c])
            weights.append(total / cnt)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"Class weights: {weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs(os.path.join("runs", "lstm"), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = eval_model(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = {
                "model_state": model.state_dict(),
                "input_size": input_size,
                "hidden": args.hidden,
                "layers": args.layers,
                "class_to_idx": class_to_idx,
                "window": args.window,
            }
            torch.save(ckpt, os.path.join("runs", "lstm", "best.pt"))
            print(f"Saved new best checkpoint (acc {best_val_acc:.3f})")


if __name__ == "__main__":
    main()
