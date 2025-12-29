"""
Simple example demonstrating basic usage of the PoseDetector class.
"""

import sys
from pathlib import Path

# Add parent directory to path to import deep_watch
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_watch import PoseDetector


def main():
    print("=" * 50)
    print("YOLOv8 Pose Detection - Simple Example")
    print("=" * 50)
    
    # Initialize the pose detector
    print("\n1. Initializing YOLOv8 Pose Detector...")
    detector = PoseDetector(
        model_path='yolov8n-pose.pt',  # Nano model for faster inference
        conf_threshold=0.5,  # Higher confidence for better accuracy
        device='cpu'  # Use 'cuda' if GPU is available
    )
    print(f"   {detector}")
    
    print("\n2. Available keypoints:")
    for i, name in enumerate(detector.keypoint_names):
        print(f"   {i}: {name}")
    
    print("\n3. Model information:")
    print(f"   - Model path: {detector.model_path}")
    print(f"   - Confidence threshold: {detector.conf_threshold}")
    print(f"   - IoU threshold: {detector.iou_threshold}")
    print(f"   - Device: {detector.model.device}")
    print(f"   - Number of keypoints: {len(detector.keypoint_names)}")
    print(f"   - Skeleton connections: {len(detector.skeleton)}")
    
    print("\n" + "=" * 50)
    print("Initialization Complete!")
    print("=" * 50)
    
    print("\nTo use the detector:")
    print("  - For images: python examples/detect_image.py --image <path>")
    print("  - For video: python examples/detect_video.py --source <path or webcam_id>")
    print("  - For webcam: python examples/detect_video.py --source 0")


if __name__ == '__main__':
    main()
