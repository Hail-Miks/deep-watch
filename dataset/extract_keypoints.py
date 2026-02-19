import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure project root is on sys.path to import deep_watch
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deep_watch import PoseDetector


def choose_person(people: List[Dict]) -> Optional[Dict]:
    """Pick one person when multiple are detected.
    Strategy: highest average keypoint confidence; tie-breaker: largest bbox area.
    """
    if not people:
        return None

    def score(p: Dict) -> Tuple[float, float]:
        kps = p.get("keypoints", {})
        confs = [kp["confidence"] for kp in kps.values()]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        bbox = p.get("bbox")
        if bbox:
            area = max(0.0, (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"]))
        else:
            area = 0.0
        return avg_conf, area

    people_sorted = sorted(people, key=score, reverse=True)
    return people_sorted[0]


def extract_sequence_from_frames(
    frame_paths: List[str], detector: PoseDetector, keypoint_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run detector on each frame path and build arrays.
    Returns (kpts[T,17,2], conf[T,17], bbox[T,4]) where missing values are NaN/0.
    """
    T = len(frame_paths)
    K = len(keypoint_names)
    kpts = np.full((T, K, 2), np.nan, dtype=np.float32)
    conf = np.zeros((T, K), dtype=np.float32)
    bbox = np.full((T, 4), np.nan, dtype=np.float32)

    for t, path in enumerate(frame_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        results, _ = detector.detect_image(img, visualize=False)
        if not results:
            continue
        result = results[0]
        people = detector.get_keypoints(result)
        person = choose_person(people)
        if person is None:
            continue
        # Fill keypoints
        for i, name in enumerate(keypoint_names):
            kp = person["keypoints"].get(name)
            if kp is None:
                continue
            kpts[t, i, 0] = kp["x"]
            kpts[t, i, 1] = kp["y"]
            conf[t, i] = kp["confidence"]
        # Fill bbox
        if person.get("bbox"):
            bb = person["bbox"]
            bbox[t] = [bb["x1"], bb["y1"], bb["x2"], bb["y2"]]

    return kpts, conf, bbox


def extract_sequence_from_video(video_path: str, detector: PoseDetector, keypoint_names: List[str]):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Run detection per frame
    T = len(frames)
    K = len(keypoint_names)
    kpts = np.full((T, K, 2), np.nan, dtype=np.float32)
    conf = np.zeros((T, K), dtype=np.float32)
    bbox = np.full((T, 4), np.nan, dtype=np.float32)

    for t, frame in enumerate(frames):
        results, _ = detector.detect_image(frame, visualize=False)
        if not results:
            continue
        result = results[0]
        people = detector.get_keypoints(result)
        person = choose_person(people)
        if person is None:
            continue
        for i, name in enumerate(keypoint_names):
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


def find_mp4s_or_frame_dirs(source_dir: str) -> List[Tuple[str, str, str]]:
    """Return list of (mode, clip_name, path).
    mode is either 'video' for .mp4 or 'frames' for a folder with jpg/png frames.
    """
    items: List[Tuple[str, str, str]] = []

    # .mp4 files
    for mp4 in sorted(glob.glob(os.path.join(source_dir, "*.mp4"))):
        clip = os.path.splitext(os.path.basename(mp4))[0]
        items.append(("video", clip, mp4))

    # frame folders: any direct subfolder containing jpg/png
    for entry in sorted(os.listdir(source_dir)):
        full = os.path.join(source_dir, entry)
        if not os.path.isdir(full):
            continue
        imgs = glob.glob(os.path.join(full, "*.jpg")) + glob.glob(os.path.join(full, "*.png"))
        if imgs:
            items.append(("frames", entry, full))

    return items


def main():
    parser = argparse.ArgumentParser(description="Extract YOLOv8 pose keypoints sequences.")
    parser.add_argument("--dataset-root", default=os.path.dirname(__file__), help="Path to dataset folder (default: dataset)")
    # parser.add_argument("--classes", nargs="*", default=["not_drowning", "drowning"], help="Subfolders to scan under dataset-root")
    parser.add_argument("--classes", nargs="*", default=["not_drowning", "drowning"], help="Subfolders to scan under dataset-root")
    parser.add_argument("--output-root", default=None, help="Where to save .npz files (default: <dataset-root>/keypoints)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.abspath(dataset_root)
    if args.output_root is None:
        output_root = os.path.join(dataset_root, "keypoints")
    else:
        output_root = args.output_root

    # Pick device
    device = None
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init detector once
    detector = PoseDetector(model_path="yolov8n-pose.pt", conf_threshold=0.25, device=device)
    keypoint_names = detector.keypoint_names

    os.makedirs(output_root, exist_ok=True)

    total = 0
    for cls in args.classes:
        src_dir = os.path.join(dataset_root, cls)
        if not os.path.isdir(src_dir):
            print(f"Skip missing class dir: {src_dir}")
            continue
        targets = find_mp4s_or_frame_dirs(src_dir)
        if not targets:
            print(f"No videos or frame folders found in {src_dir}")
            continue

        out_dir = os.path.join(output_root, cls)
        os.makedirs(out_dir, exist_ok=True)

        for mode, clip_name, path in targets:
            print(f"Processing [{cls}] {clip_name} ({mode}) ...")
            try:
                if mode == "video":
                    fps, kpts, conf, bbox = extract_sequence_from_video(path, detector, keypoint_names)
                else:
                    frame_paths = sorted(
                        glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))
                    )
                    fps = 30.0
                    kpts, conf, bbox = extract_sequence_from_frames(frame_paths, detector, keypoint_names)

                meta = {
                    "clip": clip_name,
                    "class": cls,
                    "source_path": path,
                    "mode": mode,
                    "fps": fps,
                    "num_frames": int(kpts.shape[0]),
                    "keypoints": keypoint_names,
                }

                npz_path = os.path.join(out_dir, f"{clip_name}.npz")
                json_path = os.path.join(out_dir, f"{clip_name}.json")

                np.savez_compressed(npz_path, kpts=kpts, conf=conf, bbox=bbox, fps=fps)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                print(f"Saved: {npz_path}")
                total += 1
            except Exception as e:
                print(f"Failed {clip_name}: {e}")

    print(f"Done. Saved {total} sequence(s) to {output_root}.")


if __name__ == "__main__":
    main()
