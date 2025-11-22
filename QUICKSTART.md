# Quick Start Guide - AR System

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd A3_Visual_Computing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python test_installation.py
   ```

## Step-by-Step Usage

### Step 1: Choose Camera Source

You can use either a local webcam or your Android phone as the camera source.

#### Option A: Local Webcam (Default)

Simply connect a USB webcam or use your laptop's built-in camera. No additional setup needed.

#### Option B: Android Phone Camera (Recommended for Mobility)

Use your Android phone as a wireless camera for better flexibility and mobility.

**Quick Setup - IP Webcam (Recommended):**
1. Install "IP Webcam" app from Google Play Store
2. Open the app and tap "Start Server" at the bottom
3. Note the URL displayed (e.g., `http://192.168.1.100:8080`)
4. Ensure your phone and computer are on the same WiFi network

**For detailed Android camera setup:**
```bash
python android_camera.py --setup
```

**Test your Android camera:**
```bash
python android_camera.py --method ipwebcam --url http://YOUR_PHONE_IP:8080
```

See `ANDROID_CAMERA.md` for complete Android camera documentation.

### Step 2: Generate Checkerboard Pattern

Generate a printable checkerboard:

```bash
python generate_checkerboard.py --output checkerboard.png
```

**Options:**
- `--width 9` - Number of inner corners (width)
- `--height 6` - Number of inner corners (height)
- `--square-size 25` - Square size in mm
- `--dpi 300` - Print resolution

**Print the checkerboard:**
1. Open `checkerboard.png`
2. Print at 100% scale (no fitting/scaling)
3. Verify square size with a ruler
4. Mount on cardboard or foam board

### Step 3: Calibrate Camera

Run camera calibration:

**For local webcam:**
```bash
python camera_calibration.py
```

**For Android phone camera:**
```bash
# Using IP Webcam
python camera_calibration.py --android ipwebcam --url http://YOUR_PHONE_IP:8080

# Using DroidCam
python camera_calibration.py --android droidcam --device-id 1

# Using RTSP
python camera_calibration.py --android rtsp --rtsp-url rtsp://YOUR_PHONE_IP:8554/live
```

**Instructions:**
1. Hold checkerboard in front of camera
2. Wait for green "Corners found!" message
3. Press **SPACE** to capture image
4. Capture 15-20 images from different:
   - Distances (near/far)
   - Angles (tilted left/right/up/down)
   - Positions (corners, center, edges of frame)
5. Press **ENTER** when done
6. Wait for calibration to complete

**Tips:**
- Ensure good, even lighting
- Keep checkerboard flat
- Cover entire camera field of view
- Include some extreme angles

### Step 4: Run AR System

Launch the AR application:

**For local webcam:**
```bash
python ar_system.py
```

**For Android phone camera:**
```bash
# Using IP Webcam
python ar_system.py --android ipwebcam --url http://YOUR_PHONE_IP:8080

# Using DroidCam
python ar_system.py --android droidcam --device-id 1

# Using RTSP
python ar_system.py --android rtsp --rtsp-url rtsp://YOUR_PHONE_IP:8554/live
```

**Controls:**
- **1** - Show cube only
- **2** - Show pyramid only
- **3** - Show axes only
- **4** - Show all objects
- **S** - Save performance report
- **ESC** - Exit

**What to expect:**
- Green text when checkerboard detected
- Red text when no detection
- Real-time FPS and latency display
- 3D objects rendered on checkerboard
- Position and stability information

### Step 5: Evaluate Performance (Optional)

Run performance evaluation:

**For local webcam:**
```bash
python evaluate_system.py
```

**For Android phone camera:**
```bash
# Using IP Webcam
python evaluate_system.py --android ipwebcam --url http://YOUR_PHONE_IP:8080

# Using DroidCam
python evaluate_system.py --android droidcam --device-id 1

# Using RTSP
python evaluate_system.py --android rtsp --rtsp-url rtsp://YOUR_PHONE_IP:8554/live
```

**Choose evaluation type:**
1. **Latency** - Measures processing time per frame
2. **Detection Rate** - Tests detection robustness
3. **Pose Stability** - Measures tracking accuracy
4. **All** - Runs complete evaluation suite

Results saved to:
- `evaluation_report.json` (machine-readable)
- `evaluation_report.txt` (human-readable)

## Troubleshooting

### "No calibration data found"
- Run `camera_calibration.py` first
- Check that `calibration.pkl` exists

### "No checkerboard detected"
- Ensure pattern is 9x6 inner corners
- Check lighting (avoid glare/shadows)
- Make sure pattern is flat and visible
- Try different distances from camera

### Low FPS / High latency
- Close other applications
- Use smaller checkerboard
- Reduce camera resolution
- Check CPU usage

### Unstable pose / Jittery objects
- Improve lighting conditions
- Use higher quality camera
- Recalibrate with more images
- Ensure checkerboard is rigid and flat

### Camera not found
- Check camera is connected
- Try different camera ID: `--camera 1`
- Check camera permissions
- Close other apps using camera

### Android camera issues
- Ensure phone and computer are on same WiFi network
- Verify the IP address is correct
- Check firewall isn't blocking the connection
- Try restarting the camera app on your phone
- See `ANDROID_CAMERA.md` for detailed troubleshooting

## Advanced Usage

### Custom Checkerboard Size

If using different checkerboard:

```bash
python camera_calibration.py
# Capture images with your pattern

python ar_system.py --checkerboard-width 7 --checkerboard-height 5 --square-size 0.03
```

### Multiple Cameras

To use a different local camera:

```bash
python ar_system.py --camera 1
```

To use Android phone camera, see examples in Step 4 above.

### Save Calibration for Later

The calibration file `calibration.pkl` can be reused:

```bash
# Calibrate once
python camera_calibration.py

# Use multiple times
python ar_system.py
python evaluate_system.py
python pose_estimation.py  # Test mode
```

## File Descriptions

| File | Purpose |
|------|---------|
| `test_installation.py` | Verify dependencies |
| `generate_checkerboard.py` | Create printable pattern |
| `camera_calibration.py` | Calibrate camera |
| `pose_estimation.py` | Pose estimation (can run standalone) |
| `ar_renderer.py` | 3D rendering library |
| `ar_system.py` | Main AR application |
| `android_camera.py` | Android phone camera integration |
| `evaluate_system.py` | Performance evaluation |
| `ANDROID_CAMERA.md` | Android camera setup guide |

## Performance Expectations

Typical performance on modern laptop:
- **FPS:** 30-60
- **Latency:** 15-30 ms
- **Detection Rate:** >90%
- **Stability:** <1mm variance (stationary)

## Next Steps

After getting familiar with the basic system:

1. **Experiment with rendering**
   - Modify `ar_renderer.py` to add new shapes
   - Change colors and sizes
   - Add textures

2. **Improve tracking**
   - Implement pose filtering (Kalman filter)
   - Add motion prediction
   - Handle occlusion

3. **Try different markers**
   - ArUco markers
   - Custom patterns
   - Multiple markers

4. **Optimize performance**
   - Multi-threading
   - GPU acceleration
   - Resolution tuning

## Getting Help

If you encounter issues:

1. Check this guide
2. Review README.md
3. Run `test_installation.py`
4. Check OpenCV documentation
5. Verify checkerboard pattern is correct

## Common Modifications

### Change Object Size

In `ar_system.py`, modify size parameter:

```python
frame = self.renderer.render_cube_opencv(frame, rvec, tvec, size=0.08)
```

### Add New Object Shape

In `ar_renderer.py`, add new render method:

```python
def render_sphere_opencv(self, frame, rvec, tvec, radius=0.025):
    # Your rendering code here
    pass
```

### Adjust Stability Window

In `pose_estimation.py`:

```python
self.max_history = 30  # Increase for smoother but slower response
```

Happy AR development!