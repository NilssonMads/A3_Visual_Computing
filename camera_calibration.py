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
    
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
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
        
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            display_frame = frame.copy()
            if ret:
                cv2.drawChessboardCorners(display_frame, self.checkerboard_size, 
                                         corners, ret)
                cv2.putText(display_frame, f"Corners found! Press SPACE to capture ({count}/{num_images})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No corners detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32 and ret:  # SPACE
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
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
                       choices=['ipwebcam', 'droidcam', 'rtsp'],
                       help='Use Android phone camera')
    parser.add_argument('--url', default='http://192.168.1.100:8080',
                       help='IP Webcam URL')
    parser.add_argument('--device-id', type=int, default=1,
                       help='DroidCam device ID')
    parser.add_argument('--rtsp-url', default='rtsp://192.168.1.100:8554/live',
                       help='RTSP stream URL')
    parser.add_argument('--num-images', type=int, default=15,
                       help='Number of calibration images to capture')
    
    args = parser.parse_args()
    
    print("=== Camera Calibration ===")
    print("This will calibrate your camera using a checkerboard pattern")
    print("Default: 9x6 inner corners, 25mm squares")
    
    # Setup camera
    cap = None
    if args.android_method:
        print(f"\nUsing Android camera: {args.android_method}")
        from android_camera import get_android_camera
        
        android_kwargs = {}
        if args.android_method == 'ipwebcam':
            android_kwargs['url'] = args.url
        elif args.android_method == 'droidcam':
            android_kwargs['device_id'] = args.device_id
        elif args.android_method == 'rtsp':
            android_kwargs['rtsp_url'] = args.rtsp_url
        
        cap = get_android_camera(args.android_method, **android_kwargs)
        if not cap.open():
            print("Failed to open Android camera")
            return
    
    calibrator = CameraCalibrator()
    
    # Capture images
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
