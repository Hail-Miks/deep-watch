"""
Example script for YOLOv8 pose detection on images.

This script demonstrates how to use the PoseDetector class to detect
human poses in images.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import deep_watch
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_watch import PoseDetector
import cv2


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Pose Detection on Images')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt',
                        help='YOLOv8 pose model (default: yolov8n-pose.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output image')
    parser.add_argument('--show', action='store_true',
                        help='Display the result')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Initialize pose detector
    print("Initializing YOLOv8 Pose Detector...")
    detector = PoseDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Detect poses in image
    print(f"Processing image: {args.image}")
    results, visualized = detector.detect_image(args.image, visualize=True)
    
    # Print detection information
    if len(results) > 0:
        keypoints_data = detector.get_keypoints(results[0])
        print(f"\nDetected {len(keypoints_data)} person(s)")
        
        for i, person in enumerate(keypoints_data):
            print(f"\nPerson {i+1}:")
            if person['bbox']:
                bbox = person['bbox']
                print(f"  Bounding Box: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) to ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
            
            # Print some key keypoints
            key_keypoints = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            for kp_name in key_keypoints:
                kp = person['keypoints'].get(kp_name)
                if kp and kp['confidence'] > 0.5:
                    print(f"  {kp_name}: ({kp['x']:.1f}, {kp['y']:.1f}) conf={kp['confidence']:.2f}")
    else:
        print("No persons detected")
    
    # Save output if requested
    if args.output and visualized is not None:
        cv2.imwrite(args.output, visualized)
        print(f"\nOutput saved to: {args.output}")
    
    # Display result if requested
    if args.show and visualized is not None:
        cv2.imshow('YOLOv8 Pose Detection', visualized)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
