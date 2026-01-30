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
        model_path: str = "model/yolov8n-pose.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        imgsz: int = 640
    ):
        """
        Initialize the PoseDetector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 pose model or model name (e.g., 'yolov8n-pose.pt')
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for Non-Maximum Suppression (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
            imgsz: Input image size for inference (smaller = faster, e.g., 640, 480, 320)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        
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
            imgsz=self.imgsz,
            verbose=False
        )
        
        # Visualize if requested
        visualized = None
        if visualize and len(results) > 0:
            visualized = self.visualize_results(img, results[0])
        
        return results, visualized
    
    def track(
        self,
        image: Union[str, np.ndarray],
        tracker_type: str = "bytetrack.yaml"
    ) -> Tuple[List, Optional[np.ndarray]]:
        """
        Track poses in a single image (with persistent object IDs).
        
        Args:
            image: Image path or numpy array
            tracker_type: Tracker type (e.g., "bytetrack.yaml" for ByteTrack)
            
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
        
        # Perform tracking
        results = self.model.track(
            source=img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            tracker=tracker_type,
            verbose=False,
            persist=True
        )
        
        # Visualize if requested
        visualized = None
        if len(results) > 0:
            visualized = results[0].plot()
        
        return results, visualized
    
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
    
    def __repr__(self) -> str:
        return (
            f"PoseDetector(model='{self.model_path}', "
            f"conf={self.conf_threshold}, iou={self.iou_threshold}, "
            f"device='{self.model.device}')"
        )
