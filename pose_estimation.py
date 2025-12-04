#!/usr/bin/env python3
"""
Optimized Pose Estimation Module for AR System
Performance improvements for better handling when checkerboard is not detected
"""

import cv2
import numpy as np
import time


class PoseEstimator:
    """Estimates camera pose from checkerboard detection with performance optimizations"""
    
    def __init__(self, camera_matrix, dist_coeffs, checkerboard_size=(7, 9), 
                 square_size=0.020,
                 detect_scale=0.5,  # Downscale for faster detection
                 detect_interval=2,  # Skip frames when not detected
                 adaptive_interval=True):  # Dynamically adjust interval
        """
        Initialize pose estimator with performance optimizations
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            checkerboard_size: (width, height) number of inner corners
            square_size: Size of checkerboard square in meters
            detect_scale: Scale factor for detection (0.5 = half resolution, faster)
            detect_interval: Process every Nth frame when not detecting
            adaptive_interval: Automatically adjust interval based on detection success
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Performance optimization parameters
        self.detect_scale = float(detect_scale)
        self.detect_interval = max(1, int(detect_interval))
        self.adaptive_interval = adaptive_interval
        
        # Adaptive interval state
        self.current_interval = self.detect_interval
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.max_interval = 5  # Don't skip more than every 5th frame
        
        # Frame counter for interval-based skipping
        self._frame_counter = 0
        
        # Last known good pose (for displaying when skipping)
        self.last_pose = None
        self.frames_since_detection = 0
        
        # Prepare object points (3D points in checkerboard coordinate system)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), 
                            np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                        30, 0.001)
        
        # Pose stability tracking
        self.pose_history = []
        self.max_history = 10
        
    def should_detect_this_frame(self):
        """
        Determine if detection should run on this frame
        
        Returns:
            bool: True if detection should run
        """
        self._frame_counter += 1
        
        # Always detect on first frame
        if self._frame_counter == 1:
            return True
        
        # Check if this frame should be processed based on interval
        should_process = (self._frame_counter % self.current_interval) == 0
        
        return should_process
    
    def update_adaptive_interval(self, detection_success):
        """
        Adjust detection interval based on recent success/failure
        
        Args:
            detection_success: Whether detection succeeded
        """
        if not self.adaptive_interval:
            return
        
        if detection_success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            # If consistently detecting, can reduce interval (detect more frequently)
            if self.consecutive_successes > 3 and self.current_interval > 1:
                self.current_interval = max(1, self.current_interval - 1)
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # If consistently failing, increase interval (detect less frequently)
            if self.consecutive_failures > 3 and self.current_interval < self.max_interval:
                self.current_interval = min(self.max_interval, self.current_interval + 1)
    
    def detect_and_estimate_pose(self, frame, force_detect=False):
        """
        Detect checkerboard and estimate pose with performance optimizations
        
        Args:
            frame: Input image (BGR)
            force_detect: Force detection even if interval says to skip
            
        Returns:
            Dictionary with pose data or None if detection failed/skipped
        """
        # Check if we should skip detection this frame
        if not force_detect and not self.should_detect_this_frame():
            self.frames_since_detection += 1
            # Return last known pose if available and recent
            if self.last_pose is not None and self.frames_since_detection < 10:
                return self.last_pose
            return None
        
        # Convert to grayscale once
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale for faster detection
        gray = gray_full
        scale_inv = 1.0
        if self.detect_scale > 0 and self.detect_scale < 1.0:
            gray = cv2.resize(gray_full, None, 
                            fx=self.detect_scale, 
                            fy=self.detect_scale, 
                            interpolation=cv2.INTER_AREA)
            scale_inv = 1.0 / self.detect_scale
        
        # Find checkerboard corners on scaled image
        # Use FAST_CHECK flag for faster detection (but less robust)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        if self.consecutive_failures > 2:
            # More thorough search when struggling
            flags += cv2.CALIB_CB_FILTER_QUADS
        else:
            # Faster search when working well
            flags += cv2.CALIB_CB_FAST_CHECK
        
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, flags)
        
        # Update adaptive interval based on detection result
        self.update_adaptive_interval(ret)
        
        if not ret:
            self.frames_since_detection += 1
            return None
        
        # Detection succeeded - reset counter
        self.frames_since_detection = 0
        
        # Scale corners back to full resolution if detection was on scaled image
        if scale_inv != 1.0:
            corners = corners * scale_inv
        
        # Refine corner positions on full-resolution grayscale
        corners_refined = cv2.cornerSubPix(gray_full, corners, (11, 11), (-1, -1), 
                                          self.criteria)
        
        # Estimate pose
        success, rvec, tvec = cv2.solvePnP(self.objp, corners_refined, 
                                           self.camera_matrix, self.dist_coeffs)
        
        if not success:
            return None
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Store pose in history for stability analysis
        self.pose_history.append({
            'rvec': rvec.copy(),
            'tvec': tvec.copy(),
            'timestamp': time.time()
        })
        
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        # Cache this pose
        self.last_pose = {
            'rotation_vector': rvec,
            'translation_vector': tvec,
            'rotation_matrix': rotation_matrix,
            'corners': corners_refined,
            'detected': True
        }
        
        return self.last_pose
    
    def get_pose_stability(self):
        """
        Calculate pose stability metrics
        
        Returns:
            Dictionary with stability metrics
        """
        if len(self.pose_history) < 2:
            return None
        
        # Calculate variance in translation
        tvecs = np.array([p['tvec'].flatten() for p in self.pose_history])
        translation_variance = np.var(tvecs, axis=0)
        translation_std = np.std(tvecs, axis=0)
        
        # Calculate variance in rotation
        rvecs = np.array([p['rvec'].flatten() for p in self.pose_history])
        rotation_variance = np.var(rvecs, axis=0)
        rotation_std = np.std(rvecs, axis=0)
        
        return {
            'translation_variance': translation_variance,
            'translation_std': translation_std,
            'rotation_variance': rotation_variance,
            'rotation_std': rotation_std,
            'sample_count': len(self.pose_history)
        }
    
    def get_performance_stats(self):
        """
        Get current performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'current_interval': self.current_interval,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'frames_since_detection': self.frames_since_detection,
            'has_cached_pose': self.last_pose is not None
        }
    
    def draw_axis(self, frame, rvec, tvec, length=0.05):
        """Draw 3D coordinate axes on the frame"""
        axis_points = np.float32([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, -length]
        ])
        
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, 
                                         self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int)
        
        origin = tuple(img_points[0].ravel())
        frame = cv2.line(frame, origin, tuple(img_points[1].ravel()), 
                        (0, 0, 255), 3)  # X-axis (Red)
        frame = cv2.line(frame, origin, tuple(img_points[2].ravel()), 
                        (0, 255, 0), 3)  # Y-axis (Green)
        frame = cv2.line(frame, origin, tuple(img_points[3].ravel()), 
                        (255, 0, 0), 3)  # Z-axis (Blue)
        
        return frame
    
    def draw_cube(self, frame, rvec, tvec, size=0.05):
        """Draw a 3D cube on the frame"""
        cube_points = np.float32([
            [0, 0, 0], [0, size, 0], [size, size, 0], [size, 0, 0],
            [0, 0, size], [0, size, size], [size, size, size], [size, 0, size]
        ])
        
        img_points, _ = cv2.projectPoints(cube_points, rvec, tvec,
                                         self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int).reshape(-1, 2)
        
        # Draw cube edges
        for i in range(4):
            pt1 = tuple(img_points[i])
            pt2 = tuple(img_points[(i + 1) % 4])
            frame = cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        for i in range(4, 8):
            pt1 = tuple(img_points[i])
            pt2 = tuple(img_points[4 + (i + 1) % 4])
            frame = cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        for i in range(4):
            pt1 = tuple(img_points[i])
            pt2 = tuple(img_points[i + 4])
            frame = cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        pts = img_points[4:8].reshape((-1, 1, 2))
        frame = cv2.fillPoly(frame, [pts], (0, 200, 200))
        
        return frame