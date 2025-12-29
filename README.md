# Deep Watch - YOLOv8 Pose Detection System

A powerful and easy-to-use YOLOv8-based human pose estimation system. Deep Watch provides real-time pose detection capabilities for images, videos, and webcam streams with a simple and intuitive Python API.

## Features

- 🚀 **YOLOv8 Pose Detection**: Leverages state-of-the-art YOLOv8 architecture for accurate pose estimation
- 🎯 **17 Keypoint Detection**: Detects all major body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- 📸 **Multi-Source Support**: Works with images, videos, and webcam streams
- 🎨 **Built-in Visualization**: Automatic skeleton and keypoint visualization
- ⚡ **GPU Acceleration**: Supports CUDA for faster inference
- 🔧 **Easy to Use**: Simple API with minimal configuration required

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/Hail-Miks/deep-watch.git
cd deep-watch

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies

The main dependencies are:
- ultralytics (YOLOv8)
- opencv-python
- numpy
- Pillow
- pyyaml

## Quick Start

### Basic Usage

```python
from deep_watch import PoseDetector

# Initialize the pose detector
detector = PoseDetector(
    model_path='yolov8n-pose.pt',  # nano model for faster inference
    conf_threshold=0.25,
    device='cpu'  # or 'cuda' for GPU
)

# Detect poses in an image
results, visualized = detector.detect_image('path/to/image.jpg', visualize=True)

# Get keypoint information
keypoints = detector.get_keypoints(results[0])
print(f"Detected {len(keypoints)} person(s)")
```

### Example Scripts

The repository includes several example scripts:

#### 1. Simple Initialization Example

```bash
python examples/simple_example.py
```

This script demonstrates basic initialization and displays model information.

#### 2. Image Detection

```bash
python examples/detect_image.py --image path/to/image.jpg --show
```

Options:
- `--image`: Path to input image (required)
- `--model`: YOLOv8 model to use (default: yolov8n-pose.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Path to save output image
- `--show`: Display the result
- `--device`: Device to run on (cpu or cuda)

#### 3. Video/Webcam Detection

```bash
# Process a video file
python examples/detect_video.py --source path/to/video.mp4 --output output.mp4

# Use webcam (default camera)
python examples/detect_video.py --source 0
```

Options:
- `--source`: Video file path or webcam index (default: 0)
- `--model`: YOLOv8 model to use (default: yolov8n-pose.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Path to save output video
- `--no-display`: Don't display video while processing
- `--max-frames`: Maximum number of frames to process
- `--device`: Device to run on (cpu or cuda)

## API Reference

### PoseDetector Class

#### Initialization

```python
PoseDetector(
    model_path: str = "yolov8n-pose.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: Optional[str] = None
)
```

**Parameters:**
- `model_path`: Path to YOLOv8 pose model or model name
  - Available models: yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt
- `conf_threshold`: Confidence threshold for detections (0.0-1.0)
- `iou_threshold`: IoU threshold for Non-Maximum Suppression (0.0-1.0)
- `device`: Device to run inference on ('cpu', 'cuda', or None for auto-detect)

#### Methods

##### detect_image()

```python
detect_image(
    image: Union[str, np.ndarray],
    visualize: bool = True
) -> Tuple[List, Optional[np.ndarray]]
```

Detect poses in a single image.

**Parameters:**
- `image`: Image path or numpy array
- `visualize`: Whether to create a visualization

**Returns:**
- Tuple of (results, visualized_image)

##### process_video()

```python
process_video(
    video_source: Union[str, int],
    output_path: Optional[str] = None,
    display: bool = True,
    max_frames: Optional[int] = None
) -> None
```

Process video with pose detection.

**Parameters:**
- `video_source`: Video file path or webcam index (0 for default webcam)
- `output_path`: Path to save output video (optional)
- `display`: Whether to display the video while processing
- `max_frames`: Maximum number of frames to process

##### get_keypoints()

```python
get_keypoints(result) -> List[dict]
```

Extract keypoint information from detection result.

**Returns:**
- List of dictionaries containing keypoint information for each detected person

##### visualize_results()

```python
visualize_results(
    image: np.ndarray,
    result,
    line_thickness: int = 2,
    keypoint_radius: int = 5
) -> np.ndarray
```

Visualize pose detection results on the image.

## Detected Keypoints

The system detects 17 keypoints for each person:

1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle

## Model Options

YOLOv8 offers several pose models with different sizes and performance characteristics:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n-pose.pt | Nano | Fastest | Good |
| yolov8s-pose.pt | Small | Fast | Better |
| yolov8m-pose.pt | Medium | Moderate | Good |
| yolov8l-pose.pt | Large | Slow | Better |
| yolov8x-pose.pt | Extra Large | Slowest | Best |

The model will be automatically downloaded on first use.

## GPU Support

To use GPU acceleration:

```python
detector = PoseDetector(device='cuda')
```

Make sure you have:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA installed

## Project Structure

```
deep-watch/
├── deep_watch/           # Main package
│   ├── __init__.py
│   └── pose_detector.py  # PoseDetector class
├── examples/             # Example scripts
│   ├── simple_example.py
│   ├── detect_image.py
│   └── detect_video.py
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

## Use Cases

- **Fitness Applications**: Track exercise form and count repetitions
- **Sports Analysis**: Analyze athlete movements and techniques
- **Security Systems**: Monitor person activities and detect falls
- **Healthcare**: Assist in physical therapy and rehabilitation
- **Animation**: Capture motion for character animation
- **Gaming**: Enable pose-based game controls

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Powered by PyTorch and OpenCV

## Support

For issues, questions, or suggestions, please open an issue on GitHub.