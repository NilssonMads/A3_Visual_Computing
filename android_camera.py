#!/usr/bin/env python3
"""
Android Phone Camera Integration for AR System
Supports multiple methods to stream video from Android phone to computer
"""

import cv2
import numpy as np
import socket
import struct
import pickle
import subprocess
import time
import os


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
    
    def __init__(self, url):
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
    
    def open(self):
        """Open connection to IP Webcam"""
        try:
            self.cap = cv2.VideoCapture(self.video_url)
            self.is_opened = self.cap.isOpened()
            
            if self.is_opened:
                print(f"✓ Connected to IP Webcam at {self.url}")
                
                # Get camera info
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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


class DroidCamSource(AndroidCameraSource):
    """
    Stream from DroidCam Android app
    
    Download DroidCam from Google Play Store:
    https://play.google.com/store/apps/details?id=com.dev47apps.droidcam
    
    Also install DroidCam Client on computer:
    https://www.dev47apps.com/droidcam/linux/
    
    Usage:
        1. Install DroidCam app on Android
        2. Install DroidCam Client on computer
        3. Start DroidCam on phone
        4. Connect using WiFi or USB
        5. DroidCam creates a virtual webcam device
    """
    
    def __init__(self, device_id=1):
        """
        Initialize DroidCam source
        
        Args:
            device_id: Camera device ID (DroidCam usually creates /dev/video1 or similar)
        """
        super().__init__()
        self.device_id = device_id
    
    def open(self):
        """Open DroidCam virtual camera"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            self.is_opened = self.cap.isOpened()
            
            if self.is_opened:
                print(f"✓ Connected to DroidCam on device {self.device_id}")
                
                # Get camera info
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  Resolution: {width}x{height}")
            else:
                print(f"✗ Failed to open DroidCam device {self.device_id}")
                print("  Make sure DroidCam Client is running and connected")
            
            return self.is_opened
        except Exception as e:
            print(f"✗ Error opening DroidCam: {e}")
            return False


class ADBWebcamSource(AndroidCameraSource):
    """
    Stream camera using ADB (Android Debug Bridge)
    
    Requires:
        - ADB installed on computer
        - USB debugging enabled on Android phone
        - Phone connected via USB
    
    Uses screenrecord to capture camera preview
    """
    
    def __init__(self, quality='720x1280', bitrate='8M'):
        """
        Initialize ADB webcam source
        
        Args:
            quality: Video resolution
            bitrate: Video bitrate
        """
        super().__init__()
        self.quality = quality
        self.bitrate = bitrate
        self.adb_process = None
    
    def check_adb(self):
        """Check if ADB is available and device is connected"""
        try:
            # Check if adb is installed
            result = subprocess.run(['adb', 'version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("✗ ADB not found. Please install Android Debug Bridge")
                return False
            
            # Check if device is connected
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True, timeout=5)
            
            lines = result.stdout.strip().split('\n')
            devices = [line for line in lines[1:] if line.strip() and 'device' in line]
            
            if not devices:
                print("✗ No Android device connected via ADB")
                print("  Make sure:")
                print("  1. USB debugging is enabled on phone")
                print("  2. Phone is connected via USB")
                print("  3. USB debugging permission granted")
                return False
            
            print(f"✓ ADB found, {len(devices)} device(s) connected")
            return True
            
        except FileNotFoundError:
            print("✗ ADB not found. Please install Android Debug Bridge")
            return False
        except Exception as e:
            print(f"✗ Error checking ADB: {e}")
            return False
    
    def open(self):
        """Open ADB camera stream"""
        if not self.check_adb():
            return False
        
        print("⚠ Note: ADB webcam method has limitations")
        print("  Consider using IP Webcam or DroidCam for better performance")
        
        # This is a placeholder - full ADB webcam implementation is complex
        # Would require starting camera app and capturing frames
        print("✗ ADB webcam not fully implemented")
        print("  Please use IP Webcam or DroidCam instead")
        return False


class RTSPSource(AndroidCameraSource):
    """
    Stream from phone camera using RTSP protocol
    
    Requires an app that supports RTSP streaming, such as:
    - IP Webcam (RTSP option)
    - RTSP Camera Server
    """
    
    def __init__(self, rtsp_url):
        """
        Initialize RTSP source
        
        Args:
            rtsp_url: RTSP stream URL (e.g., "rtsp://192.168.1.100:8554/live")
        """
        super().__init__()
        self.rtsp_url = rtsp_url
    
    def open(self):
        """Open RTSP stream"""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.is_opened = self.cap.isOpened()
            
            if self.is_opened:
                print(f"✓ Connected to RTSP stream at {self.rtsp_url}")
            else:
                print(f"✗ Failed to connect to RTSP stream at {self.rtsp_url}")
            
            return self.is_opened
        except Exception as e:
            print(f"✗ Error connecting to RTSP stream: {e}")
            return False


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
    
    if method == 'ipwebcam':
        url = kwargs.get('url', 'http://192.168.1.100:8080')
        return IPWebcamSource(url)
    
    elif method == 'droidcam':
        device_id = kwargs.get('device_id', 1)
        return DroidCamSource(device_id)
    
    elif method == 'rtsp':
        rtsp_url = kwargs.get('rtsp_url', 'rtsp://192.168.1.100:8554/live')
        return RTSPSource(rtsp_url)
    
    elif method == 'adb':
        return ADBWebcamSource()
    
    else:
        raise ValueError(f"Unknown method: {method}")


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
                       choices=['ipwebcam', 'droidcam', 'rtsp', 'adb'],
                       help='Connection method')
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
    
    # Test camera
    kwargs = {}
    if args.method == 'ipwebcam':
        kwargs['url'] = args.url
    elif args.method == 'droidcam':
        kwargs['device_id'] = args.device_id
    elif args.method == 'rtsp':
        kwargs['rtsp_url'] = args.rtsp_url
    
    test_android_camera(args.method, **kwargs)


if __name__ == "__main__":
    main()
