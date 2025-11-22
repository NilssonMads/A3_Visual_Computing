"""
Pose estimation module for AR system.
Estimates camera pose relative to checkerboard pattern.
"""

import cv2
import numpy as np


class PoseEstimator:
    """Estimates camera pose from checkerboard detection."""
    
    def __init__(self, camera_matrix, dist_coeffs, checkerboard_size=(9, 6), square_size=1.0):
        """
        Initialize pose estimator.
        
        Args:
            camera_matrix (np.ndarray): Camera intrinsic matrix (3x3)
            dist_coeffs (np.ndarray): Distortion coefficients
            checkerboard_size (tuple): Tuple of (width, height) internal corners
            square_size (float): Size of checkerboard square in world units
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                                     0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
    def detect_and_estimate_pose(self, frame):
        """
        Detect checkerboard and estimate camera pose.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (success, rotation_vector, translation_vector, corners)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if not ret:
            return False, None, None, None
        
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Estimate pose
        success, rvec, tvec = cv2.solvePnP(
            self.objp, 
            corners_refined, 
            self.camera_matrix, 
            self.dist_coeffs
        )
        
        return success, rvec, tvec, corners_refined
    
    def get_projection_matrix(self, rvec, tvec):
        """
        Compute projection matrix from rotation and translation vectors.
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector
            
        Returns:
            4x4 projection matrix for OpenGL
        """
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4, dtype=np.float32)
        transform_matrix[:3, :3] = rmat
        transform_matrix[:3, 3] = tvec.flatten()
        
        # OpenGL uses column-major order, so transpose
        return transform_matrix.T
    
    def draw_axis(self, frame, rvec, tvec, length=3.0):
        """
        Draw coordinate axes on the frame.
        
        Args:
            frame: Image frame to draw on
            rvec: Rotation vector
            tvec: Translation vector
            length: Length of axes
            
        Returns:
            Frame with axes drawn
        """
        # Define axis points
        axis = np.float32([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length]
        ])
        
        # Project 3D points to image plane
        img_points, _ = cv2.projectPoints(
            axis, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        img_points = img_points.astype(int)
        
        # Draw axes
        origin = tuple(img_points[0].ravel())
        frame = cv2.line(frame, origin, tuple(img_points[1].ravel()), (0, 0, 255), 3)  # X: Red
        frame = cv2.line(frame, origin, tuple(img_points[2].ravel()), (0, 255, 0), 3)  # Y: Green
        frame = cv2.line(frame, origin, tuple(img_points[3].ravel()), (255, 0, 0), 3)  # Z: Blue
        
        return frame
    
    def draw_cube(self, frame, rvec, tvec, size=3.0):
        """
        Draw a cube on the frame.
        
        Args:
            frame: Image frame to draw on
            rvec: Rotation vector
            tvec: Translation vector
            size: Size of the cube
            
        Returns:
            Frame with cube drawn
        """
        # Define cube corners
        cube_points = np.float32([
            [0, 0, 0],
            [size, 0, 0],
            [size, size, 0],
            [0, size, 0],
            [0, 0, -size],
            [size, 0, -size],
            [size, size, -size],
            [0, size, -size]
        ])
        
        # Project 3D points to image plane
        img_points, _ = cv2.projectPoints(
            cube_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        img_points = img_points.astype(int)
        
        # Draw bottom face (on checkerboard)
        for i in range(4):
            frame = cv2.line(
                frame, 
                tuple(img_points[i].ravel()), 
                tuple(img_points[(i + 1) % 4].ravel()), 
                (255, 255, 0), 
                2
            )
        
        # Draw top face
        for i in range(4, 8):
            frame = cv2.line(
                frame, 
                tuple(img_points[i].ravel()), 
                tuple(img_points[4 + (i + 1) % 4].ravel()), 
                (0, 255, 255), 
                2
            )
        
        # Draw vertical edges
        for i in range(4):
            frame = cv2.line(
                frame, 
                tuple(img_points[i].ravel()), 
                tuple(img_points[i + 4].ravel()), 
                (255, 0, 255), 
                2
            )
        
        # Fill faces with semi-transparency (optional)
        overlay = frame.copy()
        
        # Bottom face
        cv2.fillPoly(overlay, [img_points[:4]], (100, 100, 0))
        # Top face
        cv2.fillPoly(overlay, [img_points[4:8]], (0, 100, 100))
        
        # Blend
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
