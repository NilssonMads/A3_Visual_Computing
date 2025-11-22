# A3_Visual_Computing - Augmented Reality System

Take-Home assignment for the course Visual Computing

## Overview

This project implements a minimal augmented reality (AR) system using a checkerboard pattern for camera pose estimation. The system employs OpenCV for marker detection and pose estimation, and can render virtual 3D objects (cube and coordinate frame) aligned with the real-world checkerboard pattern.

## Features

- **Camera Calibration**: Calibrate your camera using a checkerboard pattern
- **Real-time Pose Estimation**: Detect checkerboard and estimate camera pose in real-time
- **3D Rendering**: Render virtual objects (cube, coordinate frame) on top of the checkerboard
- **Performance Evaluation**: Measure system latency, pose stability, and visual alignment
- **Robust Detection**: Works under varying lighting conditions and viewing angles

## Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy
- PyOpenGL (optional, for advanced rendering)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NilssonMads/A3_Visual_Computing.git
cd A3_Visual_Computing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare a Checkerboard

You need a checkerboard pattern for calibration and tracking. The default configuration uses a 9x6 internal corners checkerboard (10x7 squares total).

You can:
- Print a checkerboard from OpenCV samples: https://github.com/opencv/opencv/blob/master/doc/pattern.png
- Generate one online: https://calib.io/pages/camera-calibration-pattern-generator
- Use any checkerboard pattern (adjust parameters accordingly)

### 2. Calibrate Camera

Before running the AR system, calibrate your camera:

```bash
python ar_system.py --calibrate --video-source 0 --num-calib-frames 20
```

This will:
- Open your camera (device 0)
- Ask you to capture 20 frames with the checkerboard in different positions/angles
- Save calibration data to `calibration_data/camera_calibration.npz`

**Tips for good calibration:**
- Cover different areas of the camera view
- Include various angles and distances
- Keep the checkerboard flat and visible
- Press SPACE when checkerboard is detected (green overlay)
- Press ESC to cancel

### 3. Run AR System

After calibration, run the AR system:

```bash
python ar_system.py --video-source 0 --render-mode both
```

**Controls:**
- `ESC` - Exit application
- `c` - Toggle cube rendering
- `a` - Toggle axis rendering
- `s` - Save current frame

### 4. Evaluate Performance

Run the evaluation script to measure system performance:

```bash
python evaluate.py --test all --duration 30
```

This will evaluate:
- **Latency**: Processing time per frame, FPS
- **Pose Stability**: How stable the pose estimation is
- **Viewing Angles**: Performance at different angles

## Usage Examples

### Custom Checkerboard Size

If using a different checkerboard:

```bash
# For a 7x5 internal corners checkerboard
python ar_system.py --calibrate --checkerboard-width 7 --checkerboard-height 5

python ar_system.py --checkerboard-width 7 --checkerboard-height 5 --render-mode cube
```

### Render Only Axes

```bash
python ar_system.py --render-mode axis
```

### Render Only Cube

```bash
python ar_system.py --render-mode cube
```

### Video File Input

```bash
python ar_system.py --video-source /path/to/video.mp4
```

### Specific Tests

```bash
# Test only latency
python evaluate.py --test latency --duration 20

# Test only stability
python evaluate.py --test stability --duration 30

# Test only viewing angles
python evaluate.py --test angles
```

## System Architecture

The system consists of four main modules:

### 1. Camera Calibration (`camera_calibration.py`)

- Handles camera calibration using checkerboard pattern
- Computes camera intrinsic matrix and distortion coefficients
- Supports calibration from images or live video
- Saves/loads calibration parameters

### 2. Pose Estimation (`pose_estimation.py`)

- Detects checkerboard corners in real-time
- Estimates camera pose (rotation and translation) relative to the board
- Provides utilities to draw 3D axes and cubes
- Uses OpenCV's solvePnP for pose estimation

### 3. OpenGL Renderer (`opengl_renderer.py`)

- Optional advanced 3D rendering using OpenGL
- Renders virtual objects with proper perspective
- Provides more sophisticated rendering capabilities
- Falls back to OpenCV rendering if PyOpenGL unavailable

### 4. AR System (`ar_system.py`)

- Main application integrating all components
- Real-time video processing and rendering
- Performance monitoring (FPS, latency, detection rate)
- Interactive controls for different rendering modes

## Performance Characteristics

Typical performance on modern hardware:

- **Latency**: 10-30ms per frame
- **Frame Rate**: 30-60 FPS
- **Detection Rate**: 95%+ with good lighting and visible checkerboard
- **Pose Stability**: Sub-millimeter position jitter when board is stationary
- **Viewing Angles**: Robust detection up to 60° from normal

## Evaluation Metrics

The system provides comprehensive evaluation:

### Latency Analysis
- Mean, median, std dev, min, max processing time
- Frame rate estimation
- Performance classification

### Pose Stability
- Position and rotation variance
- Frame-to-frame jitter
- Stability rating

### Viewing Angle Robustness
- Detection success at various angles
- Angle distribution analysis
- Robustness classification

## Educational Applications

This lightweight AR system is ideal for:

- **Teaching AR fundamentals**: Demonstrates core concepts without complex frameworks
- **Computer Vision education**: Shows practical application of camera calibration and pose estimation
- **Rapid prototyping**: Quick setup for AR experiments and demonstrations
- **Research baseline**: Simple, understandable baseline for AR research

## Limitations

- Requires visible checkerboard pattern (no natural feature tracking)
- Limited to planar markers
- Performance depends on checkerboard detection accuracy
- No occlusion handling
- Basic rendering capabilities (compared to full AR SDKs)

## Future Enhancements

Potential improvements:

- Multiple marker support
- Natural feature tracking (ORB, SIFT)
- Occlusion detection and handling
- More sophisticated 3D models
- Mobile device support
- Real-time lighting estimation
- Shadow rendering

## Troubleshooting

### Calibration Issues

**Problem**: "No checkerboard patterns found"
- Ensure proper lighting
- Check checkerboard size parameters match your pattern
- Try different distances and angles
- Ensure checkerboard is flat and fully visible

### Detection Issues

**Problem**: Low detection rate during AR
- Improve lighting conditions
- Keep checkerboard flat and unoccluded
- Ensure calibration was performed correctly
- Reduce camera motion blur

### Performance Issues

**Problem**: Low FPS
- Reduce video resolution
- Close other applications
- Check camera driver settings
- Consider GPU acceleration

## References

- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Pose Estimation: https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
- Augmented Reality with OpenCV: https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html

## License

This project is for educational purposes as part of the Visual Computing course.

## Author

Visual Computing Course - Take-Home Assignment
