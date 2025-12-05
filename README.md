# A3_Visual_Computing - Minimal Augmented Reality System

Take-Home assignment for the course Visual Computing
By Mads Møllerskov Nilsson, 201808790
## Overview

This project implements a minimal augmented reality (AR) system using a checkerboard pattern for camera pose estimation. The system detects checkerboard corners in real-time, estimates the camera pose relative to the pattern, and renders virtual 3D objects (cubes, pyramids, coordinate frames) directly on top of the checkerboard.

## Features

- **Camera Calibration**: Automated camera calibration using checkerboard patterns
- **Multiple Camera Sources**: Support for local webcams and Android phones (via IP Webcam app)
- **Real-time Pose Estimation**: Efficient pose estimation with detection skipping and downscaling for reduced latency
- **3D Object Rendering**: Multiple rendering modes:
  - 3D Cube
  - 3D Pyramid
  - Coordinate axes
  - Combined visualization
- **Robust Reconnection**: Gracefully handles video connection interruptions with automatic reconnection
- **Performance Optimization**: Features including frame skipping, detection downscaling, and threaded capture

## Requirements

- Python 3.8 or higher
- Webcam, USB camera, **or Android phone with camera**
- Printed checkerboard pattern (7x9 inner corners, 20mm squares recommended)

### Python Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python >= 4.8.0
- opencv-contrib-python >= 4.8.0
- numpy >= 1.24.0
- PyOpenGL >= 3.1.7
- PyOpenGL-accelerate >= 3.1.7
- pygame >= 2.5.0

## Getting Started

### 1. Print Checkerboard Pattern

You can generate a checkerboard pattern using:
- Online generators (search "checkerboard pattern generator")
- OpenCV calibration patterns
- Default: 7x9 inner corners, 20mm square size

**Important**: The pattern should be flat and printed on rigid material for best results.

### 2. Setup Camera Source

You can use either a local webcam or your Android phone as the camera source.

#### Local Webcam (Default)

Simply connect a USB webcam or use your laptop's built-in camera. No additional setup needed.

#### Android Phone Camera (IP Webcam App)

Use your Android phone as a wireless camera.

1. Install "IP Webcam" app from Google Play Store
2. Open the app and scroll to the bottom
3. Tap "Start Server"
4. Note the URL displayed (e.g., `http://192.168.1.100:8080`)
5. Ensure your phone and computer are on the same WiFi network

Test your connection:
```bash
python android_camera.py --url http://YOUR_PHONE_IP:8080
```

### 3. Camera Calibration

Before using the AR system, you must calibrate your camera.

**For local webcam:**
```bash
python camera_calibration.py
```

**For Android phone camera (IP Webcam):**
```bash
python camera_calibration.py --android ipwebcam --url http://YOUR_PHONE_IP:8080
```

**Instructions**:
1. Position the checkerboard in front of your camera
2. Press **SPACE** when corners are detected to capture an image
3. Move the checkerboard to different positions and angles
4. Capture 15-20 images from various positions
5. Press **ENTER** when done
6. Calibration parameters will be saved to `calibration.pkl`

**Tips for good calibration**:
- Capture images from different angles (0°, 15°, 30°, 45°)
- Cover different areas of the camera frame
- Include some tilted views
- Ensure good lighting

### 4. Run the AR System

Once calibrated, run the main AR application:

```bash
# Local webcam
python ar_system.py

# Android phone with IP Webcam
python ar_system.py --android ipwebcam --url http://YOUR_PHONE_IP:8080
```

**Controls**:
- **1** - Cube rendering mode
- **2** - Pyramid rendering mode
- **3** - Coordinate axes mode
- **4** - All objects mode
- **S** - Save performance report
- **ESC** - Exit application

**Performance Tuning Options**:
- `--detect-scale`: Detection downscale factor (0.3-1.0, lower = faster, default 0.5)
- `--detect-interval`: Run detection every Nth frame (default 2)
- `--frame-skip`: Process pose/render every Nth frame (default 1)
- `--android-threaded`: Use threaded reader for Android camera
- `--android-max-width`: Resize incoming frames (0 = no resize)

Example with optimizations:
```bash
python ar_system.py --android ipwebcam --url http://192.168.1.100:8080 --detect-scale 0.6 --frame-skip 2
```

## Project Structure

```
A3_Visual_Computing/
├── android_camera.py         # Android phone camera integration
├── ar_renderer.py            # 3D object rendering (OpenCV and OpenGL)
├── ar_system.py              # Main AR application with optimization options
├── camera_calibration.py     # Camera calibration module
├── evaluate_system.py        # Performance evaluation tools
├── generate_checkerboard.py  # Checkerboard pattern generator
├── pose_estimation.py        # Pose estimation with adaptive detection
├── test_ar_system.py         # Unit tests
├── test_installation.py      # Installation verification
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── calibration.pkl           # Generated camera calibration (created after calibration)
└── checkerboard.png          # Generated checkerboard pattern
```

## Technical Details

### Camera Calibration

The system uses Zhang's calibration method implemented in OpenCV:
- Detects checkerboard corners using `cv2.findChessboardCorners()`
- Refines corner positions with sub-pixel accuracy using `cv2.cornerSubPix()`
- Computes camera intrinsic matrix and distortion coefficients using Zhang's method
- Saves calibration data for reuse

### Pose Estimation

Pose estimation pipeline:
1. Detect checkerboard corners in input frame (with optional downscaling for speed)
2. Match corners to 3D checkerboard model
3. Solve PnP problem using `cv2.solvePnP()`
4. Track pose history for stability analysis
5. Reuse last known pose on skipped frames to reduce computation

### Performance Optimizations

- **Detection Downscaling**: Run corner detection on downscaled images (`--detect-scale`)
- **Detection Interval**: Skip detection on some frames (`--detect-interval`)
- **Frame Skipping**: Process pose/rendering only on selected frames (`--frame-skip`)
- **Threaded Capture**: Non-blocking frame reads for Android cameras (`--android-threaded`)
- **Adaptive Resolution**: Downscale incoming frames for lower latency (`--android-max-width`)

### 3D Rendering

- Uses `cv2.projectPoints()` to project 3D model points to 2D image plane
- Renders objects using OpenCV drawing functions (lightweight and fast)
- Supports multiple rendering modes (cube, pyramid, axes, combined)
- Objects appear directly on top of the checkerboard pattern
