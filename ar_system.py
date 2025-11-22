#!/usr/bin/env python3
"""
Main AR application.
Integrates checkerboard detection, pose estimation, and 3D rendering.
"""

import cv2
import numpy as np
import argparse
import time
import os

from camera_calibration import CameraCalibrator
from pose_estimation import PoseEstimator
from opengl_renderer import OpenGLRenderer


class ARSystem:
    """Augmented Reality system using checkerboard marker."""
    
    def __init__(self, calibration_file=None, checkerboard_size=(9, 6), square_size=1.0):
        """
        Initialize AR system.
        
        Args:
            calibration_file (str, optional): Path to calibration file
            checkerboard_size (tuple): Tuple of (width, height) internal corners
            square_size (float): Size of checkerboard square in world units
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.calibrator = CameraCalibrator(checkerboard_size, square_size)
        
        # Load or create calibration
        if calibration_file and os.path.exists(calibration_file):
            self.calibrator.load_calibration(calibration_file)
            self.camera_matrix = self.calibrator.camera_matrix
            self.dist_coeffs = self.calibrator.dist_coeffs
        else:
            self.camera_matrix = None
            self.dist_coeffs = None
        
        self.pose_estimator = None
        self.renderer = None
        
        # Performance metrics
        self.frame_times = []
        self.pose_history = []
        
    def calibrate_camera(self, video_source=0, num_frames=20):
        """
        Calibrate camera from video.
        
        Args:
            video_source: Camera index or video file
            num_frames: Number of frames to capture
        """
        print("Starting camera calibration...")
        camera_matrix, dist_coeffs, error = self.calibrator.calibrate_from_video(
            video_source, num_frames
        )
        
        print(f"\nCalibration complete!")
        print(f"Reprojection error: {error:.4f} pixels")
        print(f"\nCamera matrix:\n{camera_matrix}")
        print(f"\nDistortion coefficients:\n{dist_coeffs}")
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Save calibration
        os.makedirs('calibration_data', exist_ok=True)
        self.calibrator.save_calibration('calibration_data/camera_calibration.npz')
        
    def run(self, video_source=0, render_mode='cube', show_fps=True):
        """
        Run AR system.
        
        Args:
            video_source: Camera index or video file
            render_mode: 'cube', 'axis', or 'both'
            show_fps: Whether to show FPS counter
        """
        if self.camera_matrix is None:
            raise ValueError("Camera not calibrated. Run calibration first.")
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(
            self.camera_matrix, 
            self.dist_coeffs,
            self.checkerboard_size,
            self.square_size
        )
        
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("\nAR System Running")
        print("=" * 50)
        print(f"Video source: {video_source}")
        print(f"Frame size: {frame_width}x{frame_height}")
        print(f"Render mode: {render_mode}")
        print(f"Checkerboard size: {self.checkerboard_size}")
        print("=" * 50)
        print("\nControls:")
        print("  ESC - Exit")
        print("  'c' - Toggle cube rendering")
        print("  'a' - Toggle axis rendering")
        print("  's' - Save current frame")
        print("=" * 50)
        
        frame_count = 0
        detected_count = 0
        show_cube = render_mode in ['cube', 'both']
        show_axis = render_mode in ['axis', 'both']
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect checkerboard and estimate pose
            success, rvec, tvec, corners = self.pose_estimator.detect_and_estimate_pose(frame)
            
            if success:
                detected_count += 1
                
                # Store pose for stability analysis
                self.pose_history.append((rvec.copy(), tvec.copy(), time.time()))
                
                # Keep only recent history
                if len(self.pose_history) > 100:
                    self.pose_history.pop(0)
                
                # Draw checkerboard corners
                cv2.drawChessboardCorners(frame, self.checkerboard_size, corners, True)
                
                # Render virtual objects
                if show_axis:
                    frame = self.pose_estimator.draw_axis(frame, rvec, tvec, length=3.0)
                
                if show_cube:
                    frame = self.pose_estimator.draw_cube(frame, rvec, tvec, size=3.0)
                
                # Display pose information
                rot_mat, _ = cv2.Rodrigues(rvec)
                cv2.putText(frame, f"Pose detected", (10, frame_height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"T: [{tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f}]",
                           (10, frame_height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "No marker detected", (10, frame_height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Calculate and display FPS
            end_time = time.time()
            frame_time = end_time - start_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            if show_fps and len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                latency_ms = avg_frame_time * 1000
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Detection: {detected_count}/{frame_count}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('AR System', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                show_cube = not show_cube
                print(f"Cube rendering: {'ON' if show_cube else 'OFF'}")
            elif key == ord('a'):
                show_axis = not show_axis
                print(f"Axis rendering: {'ON' if show_axis else 'OFF'}")
            elif key == ord('s'):
                filename = f"ar_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "=" * 50)
        print("Session Statistics")
        print("=" * 50)
        print(f"Total frames: {frame_count}")
        print(f"Detected frames: {detected_count}")
        print(f"Detection rate: {detected_count/frame_count*100:.1f}%")
        
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            print(f"Average frame time: {avg_time*1000:.2f}ms")
            print(f"Average FPS: {1.0/avg_time:.1f}")
        
        # Analyze pose stability
        if len(self.pose_history) > 1:
            self._analyze_pose_stability()
    
    def _analyze_pose_stability(self):
        """Analyze pose estimation stability."""
        print("\n" + "=" * 50)
        print("Pose Stability Analysis")
        print("=" * 50)
        
        # Calculate position variance
        positions = np.array([tvec.flatten() for _, tvec, _ in self.pose_history])
        pos_mean = np.mean(positions, axis=0)
        pos_std = np.std(positions, axis=0)
        
        print(f"Position mean: [{pos_mean[0]:.3f}, {pos_mean[1]:.3f}, {pos_mean[2]:.3f}]")
        print(f"Position std dev: [{pos_std[0]:.3f}, {pos_std[1]:.3f}, {pos_std[2]:.3f}]")
        
        # Calculate rotation variance (simplified)
        rotations = np.array([rvec.flatten() for rvec, _, _ in self.pose_history])
        rot_std = np.std(rotations, axis=0)
        print(f"Rotation std dev: [{rot_std[0]:.3f}, {rot_std[1]:.3f}, {rot_std[2]:.3f}] rad")
        
        # Calculate jitter (frame-to-frame variation)
        pos_diffs = np.diff(positions, axis=0)
        avg_jitter = np.mean(np.linalg.norm(pos_diffs, axis=1))
        print(f"Average position jitter: {avg_jitter:.4f} units/frame")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AR System with Checkerboard Tracking')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run camera calibration')
    parser.add_argument('--calibration-file', type=str,
                       default='calibration_data/camera_calibration.npz',
                       help='Path to calibration file')
    parser.add_argument('--video-source', type=int, default=0,
                       help='Video source (camera index or file)')
    parser.add_argument('--checkerboard-width', type=int, default=9,
                       help='Checkerboard internal corners width')
    parser.add_argument('--checkerboard-height', type=int, default=6,
                       help='Checkerboard internal corners height')
    parser.add_argument('--square-size', type=float, default=1.0,
                       help='Checkerboard square size in world units')
    parser.add_argument('--render-mode', type=str, default='both',
                       choices=['cube', 'axis', 'both'],
                       help='Rendering mode')
    parser.add_argument('--num-calib-frames', type=int, default=20,
                       help='Number of frames for calibration')
    
    args = parser.parse_args()
    
    # Create AR system
    checkerboard_size = (args.checkerboard_width, args.checkerboard_height)
    
    ar_system = ARSystem(
        calibration_file=args.calibration_file if not args.calibrate else None,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size
    )
    
    # Run calibration if requested
    if args.calibrate:
        ar_system.calibrate_camera(args.video_source, args.num_calib_frames)
        print("\nCalibration complete. Run again without --calibrate to start AR system.")
        return
    
    # Check if calibration exists
    if not os.path.exists(args.calibration_file):
        print(f"Error: Calibration file not found: {args.calibration_file}")
        print("Run with --calibrate flag to calibrate camera first.")
        return
    
    # Run AR system
    try:
        ar_system.run(
            video_source=args.video_source,
            render_mode=args.render_mode
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
