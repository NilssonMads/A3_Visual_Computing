#!/usr/bin/env python3
"""
Android Phone Camera Integration for AR System
Supports multiple methods to stream video from Android phone to computer
"""

import cv2
import time

import threading


class ThreadedCapture:
    """Small threaded wrapper around cv2.VideoCapture that keeps the latest frame."""

    def __init__(self, src, backend=None, max_fps=None):
        self.src = src
        self.backend = backend
        self.max_fps = max_fps
        self.cap = None
        self.thread = None
        self.running = False
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()

    def open(self):
        try:
            if self.backend is not None:
                self.cap = cv2.VideoCapture(self.src, self.backend)
            else:
                self.cap = cv2.VideoCapture(self.src)

            if not self.cap.isOpened():
                return False

            self.running = True
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
            return True
        except Exception:
            return False

    def _reader(self):
        min_delay = 0
        if self.max_fps and self.max_fps > 0:
            min_delay = 1.0 / float(self.max_fps)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            with self.lock:
                self.ret = True
                self.frame = frame

            if min_delay > 0:
                time.sleep(min_delay)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            # return a copy to avoid race conditions
            return True, self.frame.copy()

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()


class AndroidCameraSource:
    """Base class for Android camera sources"""
    
    def __init__(self):
        self.cap = None
        self.is_opened = False
    
    def open(self):
        """Open camera connection"""
        raise NotImplementedError
    
    def read(self):
        """Read frame from camera"""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
    
    def isOpened(self):
        """Check if camera is opened"""
        return self.is_opened


class IPWebcamSource(AndroidCameraSource):
    """
    Stream from IP Webcam Android app
    
    Download IP Webcam from Google Play Store:
    https://play.google.com/store/apps/details?id=com.pas.webcam
    
    Usage:
        1. Install IP Webcam app on Android
        2. Start server in the app
        3. Note the URL displayed (e.g., http://192.168.1.100:8080)
        4. Use that URL with this class
    """
    
    def __init__(self, url, threaded=False, max_width=None, max_height=None, max_fps=None):
        """
        Initialize IP Webcam source
        
        Args:
            url: URL of IP Webcam stream (e.g., "http://192.168.1.100:8080")
        """
        super().__init__()
        self.url = url
        if not url.endswith('/'):
            self.url += '/'
        self.video_url = self.url + 'video'
        # Optional performance tuning
        self.threaded = threaded
        self.max_width = max_width
        self.max_height = max_height
        self.max_fps = max_fps
    
    def open(self):
        """Open connection to IP Webcam"""
        try:
            # Optionally wrap the VideoCapture in a threaded reader to reduce latency
            if self.threaded:
                # Use a small threaded capture helper defined below
                self.cap = ThreadedCapture(self.video_url, max_fps=self.max_fps)
                self.is_opened = self.cap.open()
            else:
                self.cap = cv2.VideoCapture(self.video_url)
                self.is_opened = self.cap.isOpened()
            
            if self.is_opened:
                print(f"✓ Connected to IP Webcam at {self.url}")

                # Get camera info (support threaded wrapper)
                try:
                    if isinstance(self.cap, ThreadedCapture):
                        # underlying cv2.VideoCapture is at self.cap.cap
                        width = int(self.cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                except Exception:
                    width, height = 0, 0

                print(f"  Resolution: {width}x{height}")
            else:
                print(f"✗ Failed to connect to IP Webcam at {self.url}")
                print("  Make sure:")
                print("  1. IP Webcam app is running on your phone")
                print("  2. Phone and computer are on the same network")
                print("  3. URL is correct (check app for current IP)")
            
            return self.is_opened
        except Exception as e:
            print(f"✗ Error connecting to IP Webcam: {e}")
            return False

    def read(self):
        """Read a frame and optionally resize to max dimensions"""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Resize if requested (preserve aspect ratio)
        try:
            if self.max_width and self.max_width > 0:
                h, w = frame.shape[:2]
                if w > self.max_width:
                    scale = float(self.max_width) / float(w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            pass

        return True, frame


# Note: Only IPWebcamSource is supported in this simplified build.
# Other methods (DroidCam, RTSP, ADB) were removed to focus on IP Webcam app usage.


def get_android_camera(method='ipwebcam', **kwargs):
    """
    Get Android camera source
    
    Args:
        method: Connection method ('ipwebcam', 'droidcam', 'rtsp', 'adb')
        **kwargs: Method-specific arguments
        
    Returns:
        AndroidCameraSource instance
    """
    method = method.lower()

    if method != 'ipwebcam':
        raise ValueError("Only 'ipwebcam' method is supported in this build. Please use the IP Webcam app and provide the URL via the --url flag.")

    url = kwargs.get('url', 'http://192.168.1.100:8080')
    threaded = kwargs.get('threaded', False)
    max_width = kwargs.get('max_width', None)
    max_height = kwargs.get('max_height', None)
    max_fps = kwargs.get('max_fps', None)
    return IPWebcamSource(url, threaded=threaded, max_width=max_width, max_height=max_height, max_fps=max_fps)


def test_android_camera(method='ipwebcam', **kwargs):
    """
    Test Android camera connection
    
    Args:
        method: Connection method
        **kwargs: Method-specific arguments
    """
    print(f"\n=== Testing Android Camera ({method}) ===\n")
    
    camera = get_android_camera(method, **kwargs)
    
    if not camera.open():
        print("\n✗ Failed to open camera")
        return False
    
    print("\nPress ESC to exit test\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add info overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Method: {method}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Android Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    print(f"\n✓ Test complete. Captured {frame_count} frames at {fps:.1f} FPS")
    return True


def print_setup_instructions():
    """Print setup instructions for different methods"""
    print("""
=== Android Phone Camera Setup Guide ===

Choose one of the following methods:

┌─────────────────────────────────────────────────────────────────┐
│ Method 1: IP Webcam (RECOMMENDED - Easiest)                    │
└─────────────────────────────────────────────────────────────────┘

1. Install "IP Webcam" app from Google Play Store
2. Open the app and scroll to bottom
3. Tap "Start Server"
4. Note the URL displayed (e.g., http://192.168.1.100:8080)
5. Use this command:
   
   python ar_system.py --android ipwebcam --url http://YOUR_PHONE_IP:8080

┌─────────────────────────────────────────────────────────────────┐
│ Method 2: DroidCam (Good quality, requires client software)    │
└─────────────────────────────────────────────────────────────────┘

1. Install "DroidCam" app from Google Play Store
2. Install DroidCam Client on your computer:
   - Linux: https://www.dev47apps.com/droidcam/linux/
   - Windows: https://www.dev47apps.com/droidcam/windows/
3. Start DroidCam app on phone
4. Connect via WiFi or USB using DroidCam Client
5. DroidCam creates a virtual webcam device
6. Use this command:
   
   python ar_system.py --android droidcam --device-id 1

┌─────────────────────────────────────────────────────────────────┐
│ Method 3: RTSP Stream (Advanced)                               │
└─────────────────────────────────────────────────────────────────┘

1. Install an RTSP streaming app (e.g., "RTSP Camera Server")
2. Start RTSP server in the app
3. Note the RTSP URL (e.g., rtsp://192.168.1.100:8554/live)
4. Use this command:
   
   python ar_system.py --android rtsp --rtsp-url rtsp://YOUR_PHONE_IP:8554/live

┌─────────────────────────────────────────────────────────────────┐
│ Troubleshooting                                                 │
└─────────────────────────────────────────────────────────────────┘

Common issues:

1. "Cannot connect to camera"
   → Ensure phone and computer are on same WiFi network
   → Check firewall settings
   → Verify URL/IP address is correct

2. "Low FPS / Laggy video"
   → Reduce video quality in app settings
   → Use USB connection (DroidCam) instead of WiFi
   → Close other apps on phone

3. "Connection timeout"
   → Check phone's IP address hasn't changed
   → Disable power saving mode on phone
   → Ensure app hasn't been killed by system

═══════════════════════════════════════════════════════════════════
""")


def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Android Phone Camera for AR System')
    parser.add_argument('--method', default='ipwebcam',
                       choices=['ipwebcam'],
                       help='Connection method (IP Webcam only)')
    parser.add_argument('--url', default='http://192.168.1.100:8080',
                       help='IP Webcam URL')
    parser.add_argument('--device-id', type=int, default=1,
                       help='DroidCam device ID')
    parser.add_argument('--rtsp-url', default='rtsp://192.168.1.100:8554/live',
                       help='RTSP stream URL')
    parser.add_argument('--setup', action='store_true',
                       help='Show setup instructions')
    
    args = parser.parse_args()
    
    if args.setup:
        print_setup_instructions()
        return
    
    # Test camera (IP Webcam only)
    kwargs = {'url': args.url}
    test_android_camera('ipwebcam', **kwargs)


if __name__ == "__main__":
    main()
