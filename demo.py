#!/usr/bin/env python3
"""
Demo script to test AR system with synthetic checkerboard.
Useful for testing without a physical camera setup.
"""

import cv2
import numpy as np
import argparse
import time

from camera_calibration import CameraCalibrator
from pose_estimation import PoseEstimator


def create_synthetic_camera():
    """Create synthetic camera parameters."""
    # Typical webcam parameters (640x480 resolution)
    focal_length = 800
    cx, cy = 320, 240
    
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # No distortion for synthetic camera
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    return camera_matrix, dist_coeffs


def create_synthetic_checkerboard_view(checkerboard_size=(9, 6), square_size=30,
                                       camera_matrix=None, dist_coeffs=None,
                                       angle_x=0, angle_y=0, distance=20):
    """
    Create a synthetic view of a checkerboard.
    
    Args:
        checkerboard_size: Internal corners (width, height)
        square_size: Square size in pixels
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        angle_x, angle_y: Rotation angles in degrees
        distance: Distance from camera
        
    Returns:
        Rendered image with checkerboard
    """
    if camera_matrix is None:
        camera_matrix, dist_coeffs = create_synthetic_camera()
    
    # Create checkerboard pattern
    width, height = checkerboard_size
    board_width = (width + 1) * square_size
    board_height = (height + 1) * square_size
    
    checkerboard = np.zeros((board_height, board_width), dtype=np.uint8)
    for i in range(height + 1):
        for j in range(width + 1):
            if (i + j) % 2 == 0:
                y1 = i * square_size
                y2 = (i + 1) * square_size
                x1 = j * square_size
                x2 = (j + 1) * square_size
                checkerboard[y1:y2, x1:x2] = 255
    
    # Add border
    checkerboard = cv2.copyMakeBorder(checkerboard, square_size, square_size, 
                                     square_size, square_size,
                                     cv2.BORDER_CONSTANT, value=128)
    
    # Create 3D transformation
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
        [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]
    ])
    
    Ry = np.array([
        [np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
        [0, 1, 0],
        [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]
    ])
    
    R = Ry @ Rx
    
    # Translation
    t = np.array([[0], [0], [distance]])
    
    # Create perspective transform using camera matrix
    rvec, _ = cv2.Rodrigues(R)
    
    # Define corners of the checkerboard in 3D
    h, w = checkerboard.shape
    corners_3d = np.float32([
        [0, 0, 0],
        [w, 0, 0],
        [w, h, 0],
        [0, h, 0]
    ])
    
    # Project to 2D
    corners_2d, _ = cv2.projectPoints(corners_3d, rvec, t, camera_matrix, dist_coeffs)
    corners_2d = corners_2d.reshape(-1, 2)
    
    # Create output image
    output_size = (640, 480)
    output = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 200
    
    # Warp checkerboard to perspective
    src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src_corners, corners_2d.astype(np.float32))
    
    warped = cv2.warpPerspective(checkerboard, M, output_size)
    
    # Blend with background
    mask = (warped > 100).astype(np.uint8) * 255
    output[mask > 0] = warped[mask > 0, np.newaxis]
    
    return output


def demo_detection(num_frames=100, save_video=None):
    """
    Demonstrate checkerboard detection with synthetic views.
    
    Args:
        num_frames: Number of frames to generate
        save_video: Optional video filename to save
    """
    print("Synthetic AR System Demo")
    print("=" * 60)
    
    # Create synthetic camera
    camera_matrix, dist_coeffs = create_synthetic_camera()
    print("Created synthetic camera")
    print(f"Camera matrix:\n{camera_matrix}\n")
    
    # Create pose estimator
    checkerboard_size = (9, 6)
    pose_estimator = PoseEstimator(camera_matrix, dist_coeffs, 
                                   checkerboard_size, square_size=1.0)
    print(f"Checkerboard size: {checkerboard_size}")
    print(f"Starting demo with {num_frames} frames\n")
    
    # Video writer
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video, fourcc, 20.0, (640, 480))
        print(f"Saving to: {save_video}")
    
    # Statistics
    detected_count = 0
    processing_times = []
    
    print("Press ESC to exit, SPACE to pause\n")
    
    for i in range(num_frames):
        # Vary viewing angle
        angle_x = 20 * np.sin(i * 0.05)
        angle_y = 15 * np.cos(i * 0.03)
        distance = 25 + 5 * np.sin(i * 0.02)
        
        # Generate synthetic view
        frame = create_synthetic_checkerboard_view(
            checkerboard_size=checkerboard_size,
            square_size=30,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            angle_x=angle_x,
            angle_y=angle_y,
            distance=distance
        )
        
        # Detect and estimate pose
        start_time = time.time()
        success, rvec, tvec, corners = pose_estimator.detect_and_estimate_pose(frame)
        proc_time = (time.time() - start_time) * 1000
        
        processing_times.append(proc_time)
        
        if success:
            detected_count += 1
            
            # Draw visualization
            cv2.drawChessboardCorners(frame, checkerboard_size, corners, True)
            frame = pose_estimator.draw_axis(frame, rvec, tvec, length=3.0)
            frame = pose_estimator.draw_cube(frame, rvec, tvec, size=3.0)
            
            # Display info
            cv2.putText(frame, f"Pose Detected", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"No Detection", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display stats
        cv2.putText(frame, f"Frame: {i+1}/{num_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {1000/proc_time:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Angle: ({angle_x:.1f}, {angle_y:.1f})", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save to video
        if video_writer:
            video_writer.write(frame)
        
        # Display
        cv2.imshow('AR Demo', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            cv2.waitKey(0)
    
    # Cleanup
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Report statistics
    print("\n" + "=" * 60)
    print("Demo Statistics")
    print("=" * 60)
    print(f"Total frames: {i+1}")
    print(f"Detected frames: {detected_count}")
    print(f"Detection rate: {detected_count/(i+1)*100:.1f}%")
    print(f"Average processing time: {np.mean(processing_times):.2f}ms")
    print(f"Average FPS: {1000/np.mean(processing_times):.1f}")
    
    if save_video:
        print(f"\nVideo saved to: {save_video}")


def main():
    parser = argparse.ArgumentParser(description='AR System Demo with Synthetic Checkerboard')
    parser.add_argument('--frames', type=int, default=200,
                       help='Number of frames to generate')
    parser.add_argument('--save-video', type=str,
                       help='Save demo to video file')
    
    args = parser.parse_args()
    
    demo_detection(num_frames=args.frames, save_video=args.save_video)


if __name__ == '__main__':
    main()
