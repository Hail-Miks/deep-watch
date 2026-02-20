import os
import glob
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# COCO left/right keypoint swap indices for mirror augmentation
COCO_MIRROR_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


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


def build_features(kpts: np.ndarray, conf: np.ndarray, bbox: np.ndarray, include_vel: bool = True, include_conf: bool = True) -> np.ndarray:
    """Create per-frame feature vectors.
    - Normalize positions (34D)
    - Optionally append velocities (34D)
    - Optionally append keypoint confidence scores (17D)
    Returns array of shape (T, F)  where F = 34 + 34 + 17 = 85 by default
    """
    kpts_filled = forward_fill_nan(kpts)
    conf_filled = forward_fill_nan(conf[:, :, None])[:, :, 0] if conf is not None else None
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


class WindowedKeypointDataset(Dataset):
    def __init__(self, items: List[Dict], class_to_idx: Dict[str, int], window: int = 60, stride: int = 30, include_vel: bool = True, augment: bool = False):
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.num_features = None
        self.augment = augment
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
        if self.augment:
            x = x.copy()
            # Random Gaussian noise on features
            if np.random.rand() < 0.5:
                x += np.random.randn(*x.shape).astype(np.float32) * 0.02
            # Random mirror flip (swap left/right COCO keypoints, negate x)
            if np.random.rand() < 0.5:
                flat_pair = []
                for ki in COCO_MIRROR_IDX:
                    flat_pair.extend([2 * ki, 2 * ki + 1])
                # Swap + negate positions (cols 0:34)
                x[:, :34] = x[:, flat_pair]
                x[:, 0:34:2] *= -1
                # Swap + negate velocities (cols 34:68)
                if x.shape[1] >= 68:
                    x[:, 34:68] = x[:, [i + 34 for i in flat_pair]]
                    x[:, 34:68:2] *= -1
                # Swap confidences (cols 68:85) — no sign change
                if x.shape[1] >= 85:
                    x[:, 68:85] = x[:, [COCO_MIRROR_IDX[i] + 68 for i in range(17)]]
            # Random scale jitter
            if np.random.rand() < 0.5:
                x *= np.random.uniform(0.85, 1.15)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)  # applied after LSTM — works even with 1 layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)  # (B, T, H)
        pooled = out.mean(dim=1)  # average over all time steps (better than last-only)
        pooled = self.drop(pooled)
        logits = self.fc(pooled)
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=90)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout between LSTM and FC layer")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    p.add_argument("--class-weights", default="auto", choices=["none", "auto"], help="Use class weights in loss (auto = inverse frequency)")
    p.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing (reduces overconfidence, helps false positives)")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without val improvement)")
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
    train_ds = WindowedKeypointDataset(train_items, class_to_idx, window=args.window, stride=args.stride, include_vel=True, augment=True)
    val_ds = WindowedKeypointDataset(val_items, class_to_idx, window=args.window, stride=args.stride, include_vel=True, augment=False)

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
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=args.label_smoothing)
        print(f"Class weights: {weights}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10)

    best_val_acc = 0.0
    no_improve = 0
    os.makedirs(os.path.join("runs", "lstm"), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = eval_model(model, val_loader, device, criterion)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} | lr {lr_now:.1e}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve = 0
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
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break
        scheduler.step(va_acc)


if __name__ == "__main__":
    main()
