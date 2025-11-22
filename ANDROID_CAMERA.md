# Android Camera Quick Reference

## Overview

Use your Android phone as a wireless camera for the AR system. This provides better mobility, flexibility, and often higher quality video than laptop webcams.

## Quick Start

### 1. Choose Your Method

| Method | Difficulty | Quality | Connection | Best For |
|--------|-----------|---------|------------|----------|
| IP Webcam | ⭐ Easy | Good | WiFi | Quick setup |
| DroidCam | ⭐⭐ Medium | Excellent | WiFi/USB | Best quality |
| RTSP | ⭐⭐⭐ Advanced | Good | WiFi | Custom setups |

### 2. IP Webcam Setup (Recommended)

**On your Android phone:**
1. Install "IP Webcam" from Google Play Store
2. Open the app
3. Scroll to bottom and tap "Start Server"
4. Note the URL (e.g., `http://192.168.1.100:8080`)

**On your computer:**
```bash
# Test connection
python android_camera.py --method ipwebcam --url http://192.168.1.100:8080

# Calibrate camera
python camera_calibration.py --android ipwebcam --url http://192.168.1.100:8080

# Run AR system
python ar_system.py --android ipwebcam --url http://192.168.1.100:8080
```

### 3. DroidCam Setup

**On your Android phone:**
1. Install "DroidCam" from Google Play Store
2. Open the app
3. Note the WiFi IP or connect via USB

**On your computer:**
1. Download and install DroidCam Client
   - Linux: https://www.dev47apps.com/droidcam/linux/
   - Windows: https://www.dev47apps.com/droidcam/windows/
2. Launch DroidCam Client
3. Connect to your phone (WiFi or USB)
4. DroidCam will create a virtual camera device

**Usage:**
```bash
# Test connection (device ID usually 1 or 2)
python android_camera.py --method droidcam --device-id 1

# Calibrate camera
python camera_calibration.py --android droidcam --device-id 1

# Run AR system
python ar_system.py --android droidcam --device-id 1
```

### 4. RTSP Setup

**On your Android phone:**
1. Install an RTSP server app (e.g., "RTSP Camera Server")
2. Start the server
3. Note the RTSP URL (e.g., `rtsp://192.168.1.100:8554/live`)

**Usage:**
```bash
# Test connection
python android_camera.py --method rtsp --rtsp-url rtsp://192.168.1.100:8554/live

# Calibrate camera
python camera_calibration.py --android rtsp --rtsp-url rtsp://192.168.1.100:8554/live

# Run AR system
python ar_system.py --android rtsp --rtsp-url rtsp://192.168.1.100:8554/live
```

## Common Commands

### Complete Workflow with IP Webcam

```bash
# 1. View setup instructions
python android_camera.py --setup

# 2. Test camera connection
python android_camera.py --method ipwebcam --url http://192.168.1.100:8080

# 3. Calibrate (capture 15-20 images)
python camera_calibration.py --android ipwebcam --url http://192.168.1.100:8080

# 4. Run AR system
python ar_system.py --android ipwebcam --url http://192.168.1.100:8080
```

### With Custom Checkerboard

```bash
# Using 7x5 checkerboard with 30mm squares
python ar_system.py --android ipwebcam \
    --url http://192.168.1.100:8080 \
    --checkerboard-width 7 \
    --checkerboard-height 5 \
    --square-size 0.030
```

## Troubleshooting

### Cannot Connect to Camera

**Problem:** Connection timeout or "Cannot open camera"

**Solutions:**
1. Check phone and computer are on same WiFi network
2. Verify the IP address hasn't changed
3. Check firewall isn't blocking the connection
4. Try restarting the camera app on your phone
5. Make sure app is in foreground (not killed by Android)

**For IP Webcam:**
```bash
# Check if server is reachable
ping 192.168.1.100

# Try opening in browser
# Open http://192.168.1.100:8080 in your web browser
```

### Low FPS / Laggy Video

**Solutions:**
1. Reduce resolution in app settings (720p instead of 1080p)
2. Reduce quality/bitrate settings
3. Use USB connection (DroidCam) instead of WiFi
4. Close other apps on phone
5. Move closer to WiFi router
6. Check CPU usage on computer

**For IP Webcam:**
- In app, go to Video Preferences → Quality
- Set to 50-70% for better performance
- Lower resolution to 640x480 or 800x600

### IP Address Keeps Changing

**Solution:** Set static IP for your phone

**Android:**
1. Settings → WiFi
2. Tap your network
3. Advanced → IP settings → Static
4. Set a static IP (e.g., 192.168.1.150)

### App Closes When Phone Locks

**Solution:** Disable battery optimization

**Android:**
1. Settings → Apps
2. Find IP Webcam / DroidCam
3. Battery → Battery optimization
4. Set to "Don't optimize"

### Video is Sideways/Rotated

**Solution:**
- IP Webcam: Settings → Orientation
- DroidCam: Rotate phone to landscape
- Use landscape mode for better AR performance

## Tips for Best Results

### Camera Positioning

1. **Mount your phone securely**
   - Use a phone tripod or stand
   - Avoid hand-holding for stability
   - Keep phone in landscape orientation

2. **Distance from checkerboard**
   - Start at 30-50cm distance
   - Ensure entire checkerboard is visible
   - Avoid getting too close (causes distortion)

3. **Lighting**
   - Use even, diffuse lighting
   - Avoid direct sunlight or harsh shadows
   - Position light source behind camera

### Network Performance

1. **WiFi optimization**
   - Use 5GHz WiFi if available (less interference)
   - Keep phone and computer close to router
   - Minimize other network activity

2. **Quality settings**
   - Start with medium quality (720p)
   - Increase if performance allows
   - Lower if experiencing lag

### Battery Life

1. **Keep phone charged**
   - Connect to power while using
   - Camera and WiFi drain battery quickly
   - DroidCam USB mode charges while streaming

2. **Battery saving**
   - Reduce screen brightness
   - Disable unnecessary apps
   - Use power saving mode (if doesn't affect camera)

## Performance Comparison

| Aspect | Local Webcam | IP Webcam | DroidCam |
|--------|-------------|-----------|----------|
| Setup Time | Instant | 2 min | 5 min |
| Video Quality | Medium | Good | Excellent |
| Latency | 10-20ms | 50-100ms | 20-50ms |
| Mobility | Fixed | High | Medium |
| Power Needed | None | Phone battery | USB or battery |
| Reliability | Excellent | Good | Excellent |

## Advanced Usage

### Multiple Cameras

Switch between cameras easily:

```bash
# Use front camera
python ar_system.py --android ipwebcam --url http://192.168.1.100:8080

# Use back camera (change in app settings)
# Then run same command
```

### Recording Sessions

IP Webcam can record while streaming:
1. In app: Settings → Recording
2. Enable "Record while streaming"
3. Videos saved to phone storage

### Remote Access

Access from different network (advanced):
1. Set up port forwarding on router
2. Use your public IP address
3. Add port to firewall rules
4. **Note:** Security implications - use VPN if possible

## Getting Help

1. **View setup guide:**
   ```bash
   python android_camera.py --setup
   ```

2. **Test camera:**
   ```bash
   python android_camera.py --method ipwebcam --url http://YOUR_IP:8080
   ```

3. **Check logs:**
   - IP Webcam: Check app for errors
   - DroidCam: Check DroidCam Client logs
   - Python: Look for error messages in terminal

4. **Common fixes:**
   - Restart camera app
   - Restart computer
   - Check WiFi connection
   - Verify IP address
   - Try different quality settings

## App Download Links

- **IP Webcam:** https://play.google.com/store/apps/details?id=com.pas.webcam
- **DroidCam:** https://play.google.com/store/apps/details?id=com.dev47apps.droidcam
- **RTSP Camera Server:** https://play.google.com/store/apps/details?id=com.miv.rtspcamera

## Summary

For most users, **IP Webcam** is the recommended method:
- Easiest to set up
- No additional software needed
- Good quality
- Wireless operation
- Works on Windows, Mac, and Linux

Use **DroidCam** if you need:
- Best possible video quality
- Lower latency
- USB connection option

Use **RTSP** if you need:
- Standard protocol compatibility
- Integration with other systems
- Custom streaming setups