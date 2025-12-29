"""
Example script for YOLOv8 pose detection on video or webcam.

This script demonstrates how to use the PoseDetector class to detect
human poses in video files or webcam streams.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import deep_watch
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_watch import PoseDetector


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Pose Detection on Video/Webcam')
    parser.add_argument('--source', type=str, default='0',
                        help='Video file path or webcam index (default: 0 for webcam)')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt',
                        help='YOLOv8 pose model (default: yolov8n-pose.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display video while processing')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit (webcam index)
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Initialize pose detector
    print("Initializing YOLOv8 Pose Detector...")
    detector = PoseDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Process video
    print(f"Processing video source: {source}")
    print("Press 'q' to quit during playback")
    
    try:
        detector.process_video(
            video_source=source,
            output_path=args.output,
            display=not args.no_display,
            max_frames=args.max_frames
        )
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    
    if args.output:
        print(f"\nOutput saved to: {args.output}")


if __name__ == '__main__':
    main()
