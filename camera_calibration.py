"""
Camera calibration module for AR system.
Handles camera calibration using a checkerboard pattern.
"""

import cv2
import numpy as np
import os
import glob


class CameraCalibrator:
    """Calibrates camera using checkerboard pattern."""
    
    def __init__(self, checkerboard_size=(9, 6), square_size=1.0):
        """
        Initialize calibrator.
        
        Args:
            checkerboard_size: Tuple of (width, height) internal corners
            square_size: Size of checkerboard square in world units
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) etc.
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def calibrate_from_images(self, image_paths):
        """
        Calibrate camera from multiple checkerboard images.
        
        Args:
            image_paths: List of paths to calibration images
            
        Returns:
            Tuple of (camera_matrix, distortion_coefficients, reprojection_error)
        """
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane
        
        img_shape = None
        
        for image_path in image_paths:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if img_shape is None:
                img_shape = gray.shape[::-1]
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                obj_points.append(self.objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_refined)
        
        if len(obj_points) == 0:
            raise ValueError("No checkerboard patterns found in calibration images")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error
        
        mean_error = total_error / len(obj_points)
        
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        
        return mtx, dist, mean_error
    
    def calibrate_from_video(self, video_source=0, num_frames=20):
        """
        Calibrate camera from live video feed.
        
        Args:
            video_source: Camera index or video file path
            num_frames: Number of frames to capture for calibration
            
        Returns:
            Tuple of (camera_matrix, distortion_coefficients, reprojection_error)
        """
        cap = cv2.VideoCapture(video_source)
        
        obj_points = []
        img_points = []
        img_shape = None
        
        frames_captured = 0
        
        print(f"Calibration mode: Capture {num_frames} frames with checkerboard")
        print("Press SPACE to capture a frame, ESC to cancel")
        
        while frames_captured < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if img_shape is None:
                img_shape = gray.shape[::-1]
            
            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray, self.checkerboard_size, None
            )
            
            # Draw corners for visual feedback
            display_frame = frame.copy()
            if ret_corners:
                cv2.drawChessboardCorners(
                    display_frame, self.checkerboard_size, corners, ret_corners
                )
            
            # Display frame count
            cv2.putText(
                display_frame, 
                f"Captured: {frames_captured}/{num_frames}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            cv2.imshow('Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32 and ret_corners:  # SPACE
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                obj_points.append(self.objp)
                img_points.append(corners_refined)
                frames_captured += 1
                print(f"Frame {frames_captured} captured")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(obj_points) == 0:
            raise ValueError("No frames captured for calibration")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_shape, None, None
        )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error
        
        mean_error = total_error / len(obj_points)
        
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        
        return mtx, dist, mean_error
    
    def save_calibration(self, filename):
        """Save calibration parameters to file."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated yet")
        
        np.savez(
            filename,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            checkerboard_size=self.checkerboard_size,
            square_size=self.square_size
        )
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename):
        """Load calibration parameters from file."""
        data = np.load(filename)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        self.checkerboard_size = tuple(data['checkerboard_size'])
        self.square_size = float(data['square_size'])
        print(f"Calibration loaded from {filename}")
        
        return self.camera_matrix, self.dist_coeffs
