# Deep Watch - Integrating YOLOv8-Pose, ByteTrack, and Long Short Term Memory (LSTM) for a Real-time Drowning Detection System

Co-developing a real-time drowning detection system that integrates YOLOv8-Pose for human pose estimation, ByteTrack for multi-object tracking, and LSTM for temporal sequence analysis to distinguish normal swimming from distress behaviors in residential and public pools. Responsibilities include dataset curation from pool environments, model training and evaluation using metrics such as mAP, MOTA, precision, recall, and F1 score, and contributing to a modular system architecture with a Flask-based interface and SQL logging to support lifeguards through timely, AI-driven safety alerts.​

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

| Model           | Size        | Speed    | Accuracy |
| --------------- | ----------- | -------- | -------- |
| yolov8n-pose.pt | Nano        | Fastest  | Good     |
| yolov8s-pose.pt | Small       | Fast     | Better   |
| yolov8m-pose.pt | Medium      | Moderate | Good     |
| yolov8l-pose.pt | Large       | Slow     | Better   |
| yolov8x-pose.pt | Extra Large | Slowest  | Best     |

The model will be automatically downloaded on first use.

## GPU Support

To use GPU acceleration:

```python
detector = PoseDetector(device='cuda')
```

Make sure you have:

- NVIDIA GPU with CUDA support
- PyTorch with CUDA installed

## Acknowledgments

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Powered by PyTorch and OpenCV
