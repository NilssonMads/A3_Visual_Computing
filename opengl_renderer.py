"""
OpenGL renderer for AR system.
Renders 3D objects using OpenGL with proper camera pose.
"""

import numpy as np
import cv2

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Using OpenCV fallback rendering.")


class OpenGLRenderer:
    """Renders 3D objects using OpenGL."""
    
    def __init__(self, camera_matrix, frame_width, frame_height):
        """
        Initialize OpenGL renderer.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            frame_width: Width of video frame
            frame_height: Height of video frame
        """
        self.camera_matrix = camera_matrix
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.use_opengl = OPENGL_AVAILABLE
        
        if self.use_opengl:
            self._init_opengl()
    
    def _init_opengl(self):
        """Initialize OpenGL context."""
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
    def setup_camera(self, rvec, tvec):
        """
        Setup OpenGL camera with given pose.
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector
        """
        if not self.use_opengl:
            return
        
        # Setup projection matrix from camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        near = 0.1
        far = 100.0
        
        # Build OpenGL projection matrix
        projection = np.zeros((4, 4))
        projection[0, 0] = 2.0 * fx / self.frame_width
        projection[1, 1] = 2.0 * fy / self.frame_height
        projection[0, 2] = 1.0 - 2.0 * cx / self.frame_width
        projection[1, 2] = 2.0 * cy / self.frame_height - 1.0
        projection[2, 2] = -(far + near) / (far - near)
        projection[2, 3] = -2.0 * far * near / (far - near)
        projection[3, 2] = -1.0
        
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(projection.T.flatten())
        
        # Setup modelview matrix from pose
        rmat, _ = cv2.Rodrigues(rvec)
        
        modelview = np.eye(4)
        modelview[:3, :3] = rmat
        modelview[:3, 3] = tvec.flatten()
        
        # Invert to get view matrix
        modelview = np.linalg.inv(modelview)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(modelview.T.flatten())
    
    def render_cube(self, size=3.0, position=(1.5, 1.5, -1.5)):
        """
        Render a 3D cube.
        
        Args:
            size: Size of the cube
            position: Position of cube center (x, y, z)
        """
        if not self.use_opengl:
            return None
        
        glPushMatrix()
        glTranslatef(*position)
        
        # Draw cube faces
        glBegin(GL_QUADS)
        
        # Front face (red)
        glColor4f(1.0, 0.0, 0.0, 0.7)
        glVertex3f(-size/2, -size/2, size/2)
        glVertex3f(size/2, -size/2, size/2)
        glVertex3f(size/2, size/2, size/2)
        glVertex3f(-size/2, size/2, size/2)
        
        # Back face (green)
        glColor4f(0.0, 1.0, 0.0, 0.7)
        glVertex3f(-size/2, -size/2, -size/2)
        glVertex3f(-size/2, size/2, -size/2)
        glVertex3f(size/2, size/2, -size/2)
        glVertex3f(size/2, -size/2, -size/2)
        
        # Top face (blue)
        glColor4f(0.0, 0.0, 1.0, 0.7)
        glVertex3f(-size/2, size/2, -size/2)
        glVertex3f(-size/2, size/2, size/2)
        glVertex3f(size/2, size/2, size/2)
        glVertex3f(size/2, size/2, -size/2)
        
        # Bottom face (yellow)
        glColor4f(1.0, 1.0, 0.0, 0.7)
        glVertex3f(-size/2, -size/2, -size/2)
        glVertex3f(size/2, -size/2, -size/2)
        glVertex3f(size/2, -size/2, size/2)
        glVertex3f(-size/2, -size/2, size/2)
        
        # Right face (cyan)
        glColor4f(0.0, 1.0, 1.0, 0.7)
        glVertex3f(size/2, -size/2, -size/2)
        glVertex3f(size/2, size/2, -size/2)
        glVertex3f(size/2, size/2, size/2)
        glVertex3f(size/2, -size/2, size/2)
        
        # Left face (magenta)
        glColor4f(1.0, 0.0, 1.0, 0.7)
        glVertex3f(-size/2, -size/2, -size/2)
        glVertex3f(-size/2, -size/2, size/2)
        glVertex3f(-size/2, size/2, size/2)
        glVertex3f(-size/2, size/2, -size/2)
        
        glEnd()
        
        # Draw edges
        glLineWidth(2.0)
        glColor4f(0.0, 0.0, 0.0, 1.0)
        
        glBegin(GL_LINES)
        # Bottom edges
        glVertex3f(-size/2, -size/2, -size/2)
        glVertex3f(size/2, -size/2, -size/2)
        glVertex3f(size/2, -size/2, -size/2)
        glVertex3f(size/2, -size/2, size/2)
        glVertex3f(size/2, -size/2, size/2)
        glVertex3f(-size/2, -size/2, size/2)
        glVertex3f(-size/2, -size/2, size/2)
        glVertex3f(-size/2, -size/2, -size/2)
        
        # Top edges
        glVertex3f(-size/2, size/2, -size/2)
        glVertex3f(size/2, size/2, -size/2)
        glVertex3f(size/2, size/2, -size/2)
        glVertex3f(size/2, size/2, size/2)
        glVertex3f(size/2, size/2, size/2)
        glVertex3f(-size/2, size/2, size/2)
        glVertex3f(-size/2, size/2, size/2)
        glVertex3f(-size/2, size/2, -size/2)
        
        # Vertical edges
        glVertex3f(-size/2, -size/2, -size/2)
        glVertex3f(-size/2, size/2, -size/2)
        glVertex3f(size/2, -size/2, -size/2)
        glVertex3f(size/2, size/2, -size/2)
        glVertex3f(size/2, -size/2, size/2)
        glVertex3f(size/2, size/2, size/2)
        glVertex3f(-size/2, -size/2, size/2)
        glVertex3f(-size/2, size/2, size/2)
        glEnd()
        
        glPopMatrix()
    
    def render_coordinate_frame(self, length=3.0):
        """
        Render coordinate frame axes.
        
        Args:
            length: Length of axes
        """
        if not self.use_opengl:
            return None
        
        glLineWidth(3.0)
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(length, 0.0, 0.0)
        
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, length, 0.0)
        
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, length)
        
        glEnd()
    
    def get_rendered_frame(self):
        """
        Get the rendered frame as an image.
        
        Returns:
            Rendered frame as numpy array
        """
        if not self.use_opengl:
            return None
        
        # Read pixels from OpenGL buffer
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.frame_width, self.frame_height, 
                           GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and flip vertically
        image = np.frombuffer(data, dtype=np.uint8)
        image = image.reshape((self.frame_height, self.frame_width, 3))
        image = np.flipud(image)
        
        return image
