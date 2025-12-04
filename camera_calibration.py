#!/usr/bin/env python3
"""
Camera Calibration Module for AR System
Captures checkerboard images and computes camera intrinsic parameters
"""

import cv2
import numpy as np
import pickle
import os
import argparse
from pathlib import Path


class CameraCalibrator:
    """Handles camera calibration using checkerboard pattern"""
    
    def __init__(self, checkerboard_size=(7, 9), square_size=0.020):
        """
        Initialize calibrator
        
        Args:
            checkerboard_size: (width, height) number of inner corners
            square_size: Size of checkerboard square in meters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane
        
    def capture_calibration_images(self, num_images=20, cap=None):
        """
        Capture calibration images from camera
        
        Args:
            num_images: Number of calibration images to capture
            cap: Camera capture object (if None, uses default camera)
        """
        if cap is None:
            cap = cv2.VideoCapture(0)
            close_cap = True
        else:
            close_cap = False
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        print(f"Capturing {num_images} calibration images...")
        print("Press SPACE to capture image, ESC to cancel, ENTER when done")

        # Prepare display window and screen metrics for portrait fit
        try:
            import ctypes
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        except Exception:
            screen_w, screen_h = 1920, 1080
        display_margin = max(120, int(screen_h * 0.06))
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        
        count = 0
        frame_counter = 0
        # Keep last successful detection to avoid flashing on skipped frames
        last_found = False
        last_corners = None
        frames_since_detection = 0
        # Detection tuning: allow callers to set attributes `detect_scale` and `detect_interval`
        detect_scale = getattr(self, 'detect_scale', 1.0)
        detect_interval = max(1, int(getattr(self, 'detect_interval', 1)))

        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_counter += 1
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optionally downscale for faster detection
            gray = gray_full
            scale_inv = 1.0
            if detect_scale > 0 and detect_scale < 1.0:
                gray = cv2.resize(gray_full, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
                scale_inv = 1.0 / detect_scale

            # Run detection only on selected frames to reduce load
            if (frame_counter % detect_interval) != 0:
                # Skip actual detection this frame
                found = False
                corners = None
            else:
                # Find checkerboard corners on (possibly) scaled image
                found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
                if found and scale_inv != 1.0:
                    # Scale corner coordinates back to full resolution
                    corners = corners * scale_inv

            # Update last detection cache
            if found:
                last_found = True
                last_corners = corners
                frames_since_detection = 0
            else:
                frames_since_detection += 1
            
            display_frame = frame.copy()
            # If detection skipped but we have a recent cached detection, show it to avoid flashing
            if found:
                cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners, found)
                cv2.putText(display_frame, f"Corners found! Press SPACE to capture ({count}/{num_images})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif last_found and frames_since_detection < max(1, detect_interval * 3):
                # show cached corners
                try:
                    cv2.drawChessboardCorners(display_frame, self.checkerboard_size, last_corners, True)
                    cv2.putText(display_frame, f"Using last detection ({count}/{num_images})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
                except Exception:
                    cv2.putText(display_frame, "No corners detected",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "No corners detected",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Rotate and scale display_frame for portrait preview that fits the screen height
            try:
                h, w = display_frame.shape[:2]
                if w > h:
                    display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)
                    h, w = display_frame.shape[:2]

                max_h = max(100, screen_h - display_margin)
                max_w = max(100, screen_w - int(display_margin * 0.5))
                scale = min(1.0, float(max_w) / float(w), float(max_h) / float(h))
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    display_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    cv2.resizeWindow('Calibration', new_w, new_h)
                else:
                    cv2.resizeWindow('Calibration', w, h)

            except Exception:
                pass

            cv2.imshow('Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32 and (found or (last_found and frames_since_detection < max(1, detect_interval * 3))):  # SPACE
                # Use current corners if available, otherwise use cached
                use_corners = corners if found else last_corners
                if use_corners is not None:
                    # Refine corner positions on full-resolution grayscale
                    corners_refined = cv2.cornerSubPix(
                        gray_full, use_corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    self.obj_points.append(self.objp)
                    self.img_points.append(corners_refined)
                    count += 1
                    print(f"Image {count} captured")
            elif key == 13:  # ENTER
                break
        
        if close_cap:
            cap.release()
        cv2.destroyAllWindows()
        
        return count > 0
    
    def calibrate(self, image_size):
        """
        Calibrate camera using captured points
        
        Args:
            image_size: (width, height) of images
            
        Returns:
            Dictionary containing calibration parameters
        """
        print("Calibrating camera...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, image_size, None, None
        )
        
        if ret:
            print("Calibration successful!")
            print(f"RMS error: {ret}")
            return {
                'camera_matrix': mtx,
                'distortion_coeffs': dist,
                'rotation_vectors': rvecs,
                'translation_vectors': tvecs,
                'rms_error': ret
            }
        else:
            print("Calibration failed!")
            return None
    
    def save_calibration(self, calibration_data, filename='calibration.pkl'):
        """Save calibration data to file"""
        with open(filename, 'wb') as f:
            pickle.dump(calibration_data, f)
        print(f"Calibration saved to {filename}")
    
    @staticmethod
    def load_calibration(filename='calibration.pkl'):
        """Load calibration data from file"""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)


def main():
    """Main calibration routine"""
    parser = argparse.ArgumentParser(description='Camera Calibration for AR System')
    parser.add_argument('--android', dest='android_method',
                       choices=['ipwebcam'],
                       help='Use Android phone camera (IP Webcam)')
    parser.add_argument('--url', default='http://192.168.1.100:8080',
                       help='IP Webcam URL')
    parser.add_argument('--device-id', type=int, default=1,
                       help='DroidCam device ID')
    parser.add_argument('--rtsp-url', default='rtsp://192.168.1.100:8554/live',
                       help='RTSP stream URL')
    parser.add_argument('--num-images', type=int, default=15,
                       help='Number of calibration images to capture')
    parser.add_argument('--detect-scale', type=float, default=0.5,
                    help='Detection scale factor (0.3-1.0, lower=faster)')
    parser.add_argument('--detect-interval', type=int, default=2,
                    help='Detect every Nth frame when not detecting')
    
    args = parser.parse_args()
    
    print("=== Camera Calibration ===")
    print("This will calibrate your camera using a checkerboard pattern")
    print("Default: 7x9 inner corners, 20mm squares")
    
    # Setup camera
    cap = None
    if args.android_method:
        print(f"\nUsing Android camera: {args.android_method}")
        from android_camera import get_android_camera

        android_kwargs = {'url': args.url}
        cap = get_android_camera(args.android_method, **android_kwargs)
        if not cap.open():
            print("Failed to open Android camera")
            return
    
    calibrator = CameraCalibrator()
    
    # Capture images
    # Supply detection tuning values to the calibrator instance so capture uses them
    setattr(calibrator, 'detect_scale', args.detect_scale)
    setattr(calibrator, 'detect_interval', args.detect_interval)

    if not calibrator.capture_calibration_images(num_images=args.num_images, cap=cap):
        print("Calibration cancelled or failed")
        if cap:
            cap.release()
        return
    
    # Get image size
    if cap is None:
        test_cap = cv2.VideoCapture(0)
    else:
        test_cap = cap
    
    ret, frame = test_cap.read()
    if ret:
        h, w = frame.shape[:2]
        
        if cap is None:
            test_cap.release()
        
        # Calibrate
        calib_data = calibrator.calibrate((w, h))
        
        if calib_data:
            calibrator.save_calibration(calib_data)
            print("\nCalibration parameters:")
            print(f"Camera matrix:\n{calib_data['camera_matrix']}")
            print(f"Distortion coefficients:\n{calib_data['distortion_coeffs']}")
    else:
        print("Failed to get image size")
    
    if cap:
        cap.release()


if __name__ == "__main__":
    main()
