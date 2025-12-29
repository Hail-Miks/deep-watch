"""
YOLOv8 Pose Detector Module

This module provides a PoseDetector class for human pose estimation using YOLOv8.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Union, List, Tuple


class PoseDetector:
    """
    YOLOv8 Pose Detector for human pose estimation.
    
    This class provides methods to detect human poses in images and videos using YOLOv8.
    It supports both pre-trained models and custom trained models.
    
    Attributes:
        model: YOLOv8 pose estimation model
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device: Device to run inference on ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize the PoseDetector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 pose model or model name (e.g., 'yolov8n-pose.pt')
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for Non-Maximum Suppression (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Initialize YOLOv8 model
        print(f"Loading YOLOv8 pose model: {model_path}")
        self.model = YOLO(model_path)
        
        # Set device if specified
        if device:
            self.model.to(device)
        
        print(f"Model loaded successfully on device: {self.model.device}")
        
        # COCO keypoint names (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections for visualization
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    def detect(
        self,
        source: Union[str, np.ndarray],
        save: bool = False,
        save_dir: str = "runs/pose",
        show: bool = False,
        verbose: bool = True
    ) -> List:
        """
        Perform pose detection on an image or video.
        
        Args:
            source: Image path, video path, webcam index, or numpy array
            save: Whether to save the output
            save_dir: Directory to save results
            show: Whether to display results
            verbose: Whether to print verbose output
            
        Returns:
            List of detection results
        """
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save,
            project=save_dir,
            show=show,
            verbose=verbose
        )
        
        return results
    
    def detect_image(
        self,
        image: Union[str, np.ndarray],
        visualize: bool = True
    ) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect poses in a single image.
        
        Args:
            image: Image path or numpy array
            visualize: Whether to create a visualization
            
        Returns:
            Tuple of (results, visualized_image)
        """
        # Read image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            img = image.copy()
        
        # Perform detection
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Visualize if requested
        visualized = None
        if visualize and len(results) > 0:
            visualized = self.visualize_results(img, results[0])
        
        return results, visualized
    
    def visualize_results(
        self,
        image: np.ndarray,
        result,
        line_thickness: int = 2,
        keypoint_radius: int = 5
    ) -> np.ndarray:
        """
        Visualize pose detection results on the image.
        
        Args:
            image: Input image (numpy array)
            result: Detection result from model
            line_thickness: Thickness of skeleton lines
            keypoint_radius: Radius of keypoint circles
            
        Returns:
            Visualized image
        """
        vis_image = image.copy()
        
        # Check if keypoints are available
        if result.keypoints is None or len(result.keypoints) == 0:
            return vis_image
        
        # Get keypoints data
        keypoints = result.keypoints.xy.cpu().numpy()  # Shape: (num_people, 17, 2)
        confidences = result.keypoints.conf.cpu().numpy()  # Shape: (num_people, 17)
        
        # Draw for each detected person
        for person_kpts, person_conf in zip(keypoints, confidences):
            # Draw skeleton connections
            for connection in self.skeleton:
                start_idx, end_idx = connection
                if person_conf[start_idx] > 0.5 and person_conf[end_idx] > 0.5:
                    start_point = tuple(person_kpts[start_idx].astype(int))
                    end_point = tuple(person_kpts[end_idx].astype(int))
                    cv2.line(vis_image, start_point, end_point, (0, 255, 0), line_thickness)
            
            # Draw keypoints
            for kpt, conf in zip(person_kpts, person_conf):
                if conf > 0.5:
                    center = tuple(kpt.astype(int))
                    cv2.circle(vis_image, center, keypoint_radius, (0, 0, 255), -1)
        
        return vis_image
    
    def get_keypoints(self, result) -> List[dict]:
        """
        Extract keypoint information from detection result.
        
        Args:
            result: Detection result from model
            
        Returns:
            List of dictionaries containing keypoint information for each person
        """
        if result.keypoints is None or len(result.keypoints) == 0:
            return []
        
        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf.cpu().numpy()
        
        people = []
        for person_idx, (person_kpts, person_conf) in enumerate(zip(keypoints, confidences)):
            person_data = {
                'keypoints': {},
                'bbox': None
            }
            
            # Extract keypoint coordinates and confidences
            for i, (kpt, conf) in enumerate(zip(person_kpts, person_conf)):
                person_data['keypoints'][self.keypoint_names[i]] = {
                    'x': float(kpt[0]),
                    'y': float(kpt[1]),
                    'confidence': float(conf)
                }
            
            # Add bounding box if available
            if result.boxes is not None and len(result.boxes) > person_idx:
                bbox = result.boxes.xyxy.cpu().numpy()[person_idx]
                person_data['bbox'] = {
                    'x1': float(bbox[0]),
                    'y1': float(bbox[1]),
                    'x2': float(bbox[2]),
                    'y2': float(bbox[3])
                }
            
            people.append(person_data)
        
        return people
    
    def process_video(
        self,
        video_source: Union[str, int],
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None
    ) -> None:
        """
        Process video with pose detection.
        
        Args:
            video_source: Video file path or webcam index (0 for default webcam)
            output_path: Path to save output video (optional)
            display: Whether to display the video while processing
            max_frames: Maximum number of frames to process (None for all)
        """
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check max frames limit
                if max_frames and frame_count >= max_frames:
                    break
                
                # Perform pose detection
                results, vis_frame = self.detect_image(frame, visualize=True)
                
                # Write frame if output path is specified
                if writer and vis_frame is not None:
                    writer.write(vis_frame)
                
                # Display frame
                if display and vis_frame is not None:
                    cv2.imshow('YOLOv8 Pose Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"Finished processing {frame_count} frames")
    
    def __repr__(self) -> str:
        return (
            f"PoseDetector(model='{self.model_path}', "
            f"conf={self.conf_threshold}, iou={self.iou_threshold}, "
            f"device='{self.model.device}')"
        )
