#!/usr/bin/env python3
"""
AR Renderer Module using OpenGL
Renders 3D objects using OpenGL based on pose estimation
"""

import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
from pygame.locals import *


class ARRenderer:
    """Renders AR content using OpenGL"""
    
    def __init__(self, camera_matrix, dist_coeffs, width, height):
        """
        Initialize AR renderer
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            width: Frame width
            height: Frame height
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.width = width
        self.height = height
        
        # Initialize Pygame and OpenGL
        pygame.init()
        self.display = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("AR Renderer")
        
        # Setup OpenGL
        self._setup_opengl()
        
    def _setup_opengl(self):
        """Setup OpenGL parameters"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light setup
        glLight(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLight(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1))
        
    def set_projection_from_camera(self):
        """Set OpenGL projection matrix from camera intrinsics"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Extract camera parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Set up projection matrix
        near = 0.01
        far = 100.0
        
        # Create projection matrix from camera parameters
        projection = np.zeros((4, 4))
        projection[0, 0] = 2.0 * fx / self.width
        projection[1, 1] = 2.0 * fy / self.height
        projection[0, 2] = 1.0 - 2.0 * cx / self.width
        projection[1, 2] = 2.0 * cy / self.height - 1.0
        projection[2, 2] = -(far + near) / (far - near)
        projection[2, 3] = -2.0 * far * near / (far - near)
        projection[3, 2] = -1.0
        
        glLoadMatrixf(projection.T)
        
    def set_modelview_from_pose(self, rvec, tvec):
        """
        Set OpenGL modelview matrix from pose
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector
        """
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = tvec.flatten()
        
        # OpenGL uses column-major order, OpenCV uses row-major
        # We need to transpose and invert for proper camera pose
        view_matrix = np.linalg.inv(transformation)
        
        glLoadMatrixf(view_matrix.T)
        
    def draw_cube(self, size=0.05, wireframe=False):
        """
        Draw a 3D cube
        
        Args:
            size: Size of cube
            wireframe: Draw as wireframe if True
        """
        vertices = [
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],  # Bottom
            [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]  # Top
        ]
        
        faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [2, 3, 7, 6],  # Back
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5]   # Right
        ]
        
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1]   # Cyan
        ]
        
        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_QUADS)
        for i, face in enumerate(faces):
            glColor3fv(colors[i])
            for vertex_idx in face:
                glVertex3fv(vertices[vertex_idx])
        glEnd()
        
    def draw_pyramid(self, size=0.05):
        """Draw a 3D pyramid"""
        vertices = [
            [0, 0, 0],
            [size, 0, 0],
            [size, size, 0],
            [0, size, 0],
            [size/2, size/2, size]  # Apex
        ]
        
        glBegin(GL_TRIANGLES)
        
        # Bottom face
        glColor3f(0.5, 0.5, 0.5)
        glVertex3fv(vertices[0])
        glVertex3fv(vertices[1])
        glVertex3fv(vertices[2])
        
        glVertex3fv(vertices[0])
        glVertex3fv(vertices[2])
        glVertex3fv(vertices[3])
        
        # Side faces
        glColor3f(1, 0, 0)
        glVertex3fv(vertices[0])
        glVertex3fv(vertices[1])
        glVertex3fv(vertices[4])
        
        glColor3f(0, 1, 0)
        glVertex3fv(vertices[1])
        glVertex3fv(vertices[2])
        glVertex3fv(vertices[4])
        
        glColor3f(0, 0, 1)
        glVertex3fv(vertices[2])
        glVertex3fv(vertices[3])
        glVertex3fv(vertices[4])
        
        glColor3f(1, 1, 0)
        glVertex3fv(vertices[3])
        glVertex3fv(vertices[0])
        glVertex3fv(vertices[4])
        
        glEnd()
        
    def draw_coordinate_frame(self, length=0.05):
        """Draw 3D coordinate frame (axes)"""
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X-axis (Red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        
        # Y-axis (Green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)
        
        # Z-axis (Blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        
        glEnd()
        glLineWidth(1.0)
        
    def render_background(self, frame):
        """
        Render camera frame as background
        
        Args:
            frame: Camera frame (BGR)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)  # Flip vertically for OpenGL
        
        # Disable depth test for background
        glDisable(GL_DEPTH_TEST)
        
        # Draw background as textured quad
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Create texture from frame
        glRasterPos2i(0, 0)
        glDrawPixels(self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        
        glEnable(GL_DEPTH_TEST)
        
    def clear(self):
        """Clear buffers"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
    def swap_buffers(self):
        """Swap display buffers"""
        pygame.display.flip()
        

class SimpleARRenderer:
    """Simpler AR renderer using OpenCV only (no OpenGL window)"""
    
    def __init__(self, camera_matrix, dist_coeffs):
        """
        Initialize simple AR renderer
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def render_cube_opencv(self, frame, rvec, tvec, size=0.05):
        """
        Render cube using OpenCV drawing functions
        
        Args:
            frame: Input frame
            rvec: Rotation vector
            tvec: Translation vector
            size: Cube size
            
        Returns:
            Frame with rendered cube
        """
        # Define cube vertices
        cube_points = np.float32([
            [0, 0, 0], [0, size, 0], [size, size, 0], [size, 0, 0],  # Bottom
            [0, 0, size], [0, size, size], [size, size, size], [size, 0, size]  # Top
        ])
        
        # Project to image plane
        img_points, _ = cv2.projectPoints(cube_points, rvec, tvec,
                                         self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int).reshape(-1, 2)
        
        # Draw edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
        ]
        
        for i, j in edges:
            pt1 = tuple(img_points[i])
            pt2 = tuple(img_points[j])
            frame = cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Fill faces with transparency
        # Bottom face
        pts = img_points[:4].reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (100, 100, 100))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Top face
        pts = img_points[4:8].reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 200))
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        return frame
    
    def render_pyramid_opencv(self, frame, rvec, tvec, size=0.05):
        """
        Render pyramid using OpenCV
        
        Args:
            frame: Input frame
            rvec: Rotation vector
            tvec: Translation vector
            size: Pyramid size
            
        Returns:
            Frame with rendered pyramid
        """
        # Define pyramid vertices
        pyramid_points = np.float32([
            [0, 0, 0],
            [size, 0, 0],
            [size, size, 0],
            [0, size, 0],
            [size/2, size/2, size]  # Apex
        ])
        
        # Project to image plane
        img_points, _ = cv2.projectPoints(pyramid_points, rvec, tvec,
                                         self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int).reshape(-1, 2)
        
        # Draw edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Base
            (0, 4), (1, 4), (2, 4), (3, 4)   # To apex
        ]
        
        for i, j in edges:
            pt1 = tuple(img_points[i])
            pt2 = tuple(img_points[j])
            frame = cv2.line(frame, pt1, pt2, (255, 200, 0), 2)
        
        # Fill faces
        faces = [
            ([0, 1, 4], (255, 0, 0)),
            ([1, 2, 4], (0, 255, 0)),
            ([2, 3, 4], (0, 0, 255)),
            ([3, 0, 4], (255, 255, 0))
        ]
        
        overlay = frame.copy()
        for face_indices, color in faces:
            pts = img_points[face_indices].reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
        
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        return frame
    
    def render_axes(self, frame, rvec, tvec, length=0.05):
        """
        Render coordinate axes
        
        Args:
            frame: Input frame
            rvec: Rotation vector
            tvec: Translation vector
            length: Axis length
            
        Returns:
            Frame with rendered axes
        """
        axis_points = np.float32([
            [0, 0, 0],
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length]
        ])
        
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec,
                                         self.camera_matrix, self.dist_coeffs)
        img_points = img_points.astype(int)
        
        origin = tuple(img_points[0].ravel())
        frame = cv2.line(frame, origin, tuple(img_points[1].ravel()), (0, 0, 255), 3)
        frame = cv2.line(frame, origin, tuple(img_points[2].ravel()), (0, 255, 0), 3)
        frame = cv2.line(frame, origin, tuple(img_points[3].ravel()), (255, 0, 0), 3)
        
        return frame
