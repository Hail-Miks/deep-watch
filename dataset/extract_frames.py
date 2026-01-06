import os
import cv2

"""
Extract frames from each .mp4 clip in dataset/drowning.
For a video named clip_1.mp4, frames will be saved to:
dataset/drowning/clip_1/frame_0001.jpg, frame_0002.jpg, ...
"""

def extract_frames_from_video(video_file_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_file_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")


def main():
    dataset_dir = os.path.dirname(__file__)
    source_dir = os.path.join(dataset_dir, "drowning")

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return

    videos = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(".mp4")
    ]

    if not videos:
        print(f"No .mp4 files found in {source_dir}")
        return

    print(f"Found {len(videos)} video(s) in {source_dir}")
    for vid in sorted(videos):
        video_path = os.path.join(source_dir, vid)
        name_without_ext = os.path.splitext(vid)[0]
        output_dir = os.path.join(source_dir, name_without_ext)
        print(f"Processing {vid} -> {name_without_ext}/")
        extract_frames_from_video(video_path, output_dir)


if __name__ == "__main__":
    main()
