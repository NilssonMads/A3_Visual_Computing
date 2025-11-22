# AR System - Quick Start Guide

This guide will help you get started with the AR system quickly.

## Prerequisites

- Python 3.7 or higher
- Webcam (for live AR) or video file
- Printed checkerboard pattern (or use the generator)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Step-by-Step Tutorial

### Step 1: Generate a Checkerboard Pattern

First, generate a checkerboard pattern to print:

```bash
python generate_checkerboard.py --width 9 --height 6 --square-size 60 --output my_checkerboard.png
```

This creates a 9x6 (internal corners) checkerboard image. Print this on A4 paper.

### Step 2: Verify Installation

Run the test suite to ensure everything is working:

```bash
python test_ar_system.py
```

You should see all tests passing.

### Step 3: Camera Calibration

Calibrate your camera using the printed checkerboard:

```bash
python ar_system.py --calibrate --video-source 0 --num-calib-frames 20
```

**Important calibration tips:**
1. Hold the checkerboard flat and ensure it's fully visible
2. Move it to different positions: corners, center, edges
3. Include different angles (but keep it visible)
4. Include different distances from the camera
5. Press SPACE when the checkerboard is detected (green overlay appears)
6. Capture at least 15-20 good frames
7. Press ESC when done

The calibration will be saved to `calibration_data/camera_calibration.npz`.

### Step 4: Run the AR System

Now run the AR system:

```bash
python ar_system.py --video-source 0 --render-mode both
```

You should see:
- Real-time video from your camera
- When the checkerboard is visible: 3D axes and a cube rendered on top
- FPS and latency information
- Detection statistics

**Controls:**
- `ESC` - Exit
- `c` - Toggle cube rendering
- `a` - Toggle axis rendering  
- `s` - Save current frame

### Step 5: Try the Demo (No Camera Required)

If you don't have a camera or printed checkerboard, try the synthetic demo:

```bash
python demo.py --frames 100
```

This generates synthetic checkerboard views and demonstrates the AR system.

### Step 6: Evaluate System Performance

Run comprehensive evaluation:

```bash
python evaluate.py --test all --duration 30
```

This will:
1. Measure latency (30 seconds of continuous operation)
2. Measure pose stability (keep board still for 30 seconds)
3. Test viewing angles (move camera around, press 'r' to record samples)

Results are saved to `evaluation_results.json`.

## Common Issues and Solutions

### Issue: "No checkerboard patterns found"

**Solutions:**
- Ensure good lighting (avoid shadows, glare)
- Check that the checkerboard size matches your printed pattern
- Make sure the entire checkerboard is visible
- Try different distances from the camera
- Ensure the checkerboard is flat

### Issue: "Calibration file not found"

**Solution:**
Run calibration first:
```bash
python ar_system.py --calibrate
```

### Issue: Low detection rate

**Solutions:**
- Improve lighting conditions
- Ensure the checkerboard is clean and flat
- Reduce camera motion
- Check that calibration was successful
- Use a larger checkerboard pattern

### Issue: Jittery rendering

**Solutions:**
- Ensure stable camera position
- Better lighting (reduces image noise)
- Re-calibrate camera
- Use higher quality camera

## Advanced Usage

### Custom Checkerboard Size

If you have a different checkerboard (e.g., 7x5 internal corners):

```bash
# Calibration
python ar_system.py --calibrate --checkerboard-width 7 --checkerboard-height 5

# Running
python ar_system.py --checkerboard-width 7 --checkerboard-height 5
```

### Save AR Session to Video

You can use screen recording software, or modify the code to save output.

### Batch Evaluation

Run specific tests:

```bash
# Only latency test
python evaluate.py --test latency --duration 20

# Only stability test  
python evaluate.py --test stability --duration 30

# Only viewing angle test
python evaluate.py --test angles
```

### Using Video Files

Instead of live camera:

```bash
python ar_system.py --video-source /path/to/video.mp4
```

## Understanding the Output

### AR System Output

- **FPS**: Frames per second (higher is better, aim for 30+)
- **Latency**: Processing time per frame in milliseconds (lower is better)
- **Detection Rate**: Percentage of frames where checkerboard was detected

### Evaluation Results

**Latency Metrics:**
- Mean latency: Average processing time
- 95th percentile: Maximum latency for 95% of frames
- FPS: Estimated frame rate

**Performance Ratings:**
- EXCELLENT: >60 FPS capable (<16.67ms latency)
- GOOD: 30-60 FPS (16.67-33.33ms)
- ACCEPTABLE: 20-30 FPS (33.33-50ms)
- POOR: <20 FPS (>50ms)

**Stability Metrics:**
- Position std dev: How much the estimated position varies (lower is better)
- Rotation std dev: How much the estimated rotation varies (lower is better)
- Jitter: Frame-to-frame variation (lower is better)

**Stability Ratings:**
- EXCELLENT: Very stable, sub-centimeter precision
- GOOD: Stable, minor jitter
- ACCEPTABLE: Some visible jitter
- POOR: Unstable, significant jitter

## Tips for Best Results

1. **Lighting**: Use even, bright lighting. Avoid direct sunlight or harsh shadows.

2. **Checkerboard Quality**: 
   - Print on flat, rigid paper
   - Ensure high contrast (pure black and white)
   - Keep it clean and unwrinkled

3. **Calibration**:
   - Use same lighting as you'll use for AR
   - Cover entire camera field of view
   - Include various angles and distances
   - Minimum 15 frames, 20-30 recommended

4. **Camera**:
   - Use a camera with good low-light performance
   - Disable auto-focus if possible (use fixed focus)
   - Higher resolution cameras work better

5. **Environment**:
   - Minimize camera shake
   - Avoid moving objects in background
   - Use non-reflective surfaces

## Next Steps

After getting familiar with the basic system, you can:

1. **Modify rendering**: Edit `pose_estimation.py` to render different objects
2. **Add multiple markers**: Extend to track multiple checkerboards
3. **Improve performance**: Optimize detection parameters
4. **Add features**: Implement occlusion detection, better lighting handling
5. **Port to mobile**: Adapt for Android/iOS using similar libraries

## Technical Details

- **Camera Model**: Pinhole camera model with radial distortion correction
- **Calibration Method**: Zhang's method (implemented in OpenCV)
- **Pose Estimation**: Perspective-n-Point (PnP) algorithm
- **Detection**: Checkerboard corner detection with sub-pixel refinement
- **Rendering**: OpenCV drawing functions (with optional OpenGL support)

## Resources

- OpenCV Documentation: https://docs.opencv.org/
- Camera Calibration Tutorial: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Pose Estimation Tutorial: https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
- Checkerboard Pattern Generator: https://calib.io/pages/camera-calibration-pattern-generator

## Support

For issues or questions:
1. Check the README.md for detailed information
2. Review this guide for common solutions
3. Run `python test_ar_system.py` to verify installation
4. Check that all dependencies are installed correctly
