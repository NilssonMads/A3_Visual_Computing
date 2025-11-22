#!/usr/bin/env python3
"""
Pose Estimation Module for AR System
Detects checkerboard and estimates camera pose relative to it
"""

import cv2
import numpy as np
import time


class PoseEstimator:
    """Estimates camera pose from checkerboard detection"""
    
    def __init__(self, camera_matrix, dist_coeffs, checkerboard_size=(9, 6), 
                 square_size=0.025):
        """
        Initialize pose estimator
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            checkerboard_size: (width, height) number of inner corners
            square_size: Size of checkerboard square in meters
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
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
        
    def detect_and_estimate_pose(self, frame):
        """
        Detect checkerboard and estimate pose
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            Dictionary with pose data or None if detection failed
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if not ret:
            return None
        
        # Refine corner positions
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
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
        
        return {
            'rotation_vector': rvec,
            'translation_vector': tvec,
            'rotation_matrix': rotation_matrix,
            'corners': corners_refined,
            'detected': True
        }
    
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
    
    def draw_axis(self, frame, rvec, tvec, length=0.05):
        """
        Draw 3D coordinate axes on the frame
        
        Args:
            frame: Input image
            rvec: Rotation vector
            tvec: Translation vector
            length: Length of axes in meters
            
        Returns:
            Frame with axes drawn
        """
        # Define 3D points for axes
        axis_points = np.float32([
            [0, 0, 0],           # Origin
            [length, 0, 0],      # X-axis
            [0, length, 0],      # Y-axis
            [0, 0, length]       # Z-axis
        ])
        
        # Project 3D points to image plane
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, 
                                         self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int)
        
        # Draw axes
        origin = tuple(img_points[0].ravel())
        frame = cv2.line(frame, origin, tuple(img_points[1].ravel()), 
                        (0, 0, 255), 3)  # X-axis (Red)
        frame = cv2.line(frame, origin, tuple(img_points[2].ravel()), 
                        (0, 255, 0), 3)  # Y-axis (Green)
        frame = cv2.line(frame, origin, tuple(img_points[3].ravel()), 
                        (255, 0, 0), 3)  # Z-axis (Blue)
        
        return frame
    
    def draw_cube(self, frame, rvec, tvec, size=0.05):
        """
        Draw a 3D cube on the frame
        
        Args:
            frame: Input image
            rvec: Rotation vector
            tvec: Translation vector
            size: Size of cube in meters
            
        Returns:
            Frame with cube drawn
        """
        # Define 3D points for cube vertices
        cube_points = np.float32([
            [0, 0, 0], [0, size, 0], [size, size, 0], [size, 0, 0],  # Bottom
            [0, 0, size], [0, size, size], [size, size, size], [size, 0, size]  # Top
        ])
        
        # Project 3D points to image plane
        img_points, _ = cv2.projectPoints(cube_points, rvec, tvec,
                                         self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int).reshape(-1, 2)
        
        # Draw cube edges
        # Bottom face
        for i in range(4):
            pt1 = tuple(img_points[i])
            pt2 = tuple(img_points[(i + 1) % 4])
            frame = cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Top face
        for i in range(4, 8):
            pt1 = tuple(img_points[i])
            pt2 = tuple(img_points[4 + (i + 1) % 4])
            frame = cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Vertical edges
        for i in range(4):
            pt1 = tuple(img_points[i])
            pt2 = tuple(img_points[i + 4])
            frame = cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Fill top face for better visibility
        pts = img_points[4:8].reshape((-1, 1, 2))
        frame = cv2.fillPoly(frame, [pts], (0, 200, 200))
        
        return frame


def test_pose_estimation():
    """Test pose estimation with live camera"""
    from camera_calibration import CameraCalibrator
    
    # Load calibration data
    calib_data = CameraCalibrator.load_calibration('calibration.pkl')
    
    if calib_data is None:
        print("No calibration data found. Please run camera_calibration.py first")
        return
    
    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['distortion_coeffs']
    
    # Initialize pose estimator
    estimator = PoseEstimator(camera_matrix, dist_coeffs)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    print("Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Estimate pose
        pose_data = estimator.detect_and_estimate_pose(frame)
        
        if pose_data:
            # Draw coordinate axes
            frame = estimator.draw_axis(frame, pose_data['rotation_vector'],
                                       pose_data['translation_vector'])
            
            # Draw cube
            frame = estimator.draw_cube(frame, pose_data['rotation_vector'],
                                       pose_data['translation_vector'], size=0.05)
            
            # Display pose info
            tvec = pose_data['translation_vector'].flatten()
            cv2.putText(frame, f"Position: ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})m",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Get stability metrics
            stability = estimator.get_pose_stability()
            if stability:
                std = stability['translation_std']
                cv2.putText(frame, f"Stability (std): ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "No checkerboard detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Pose Estimation Test', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pose_estimation()
