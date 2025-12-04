#!/usr/bin/env python3
"""
Main AR Application
Combines pose estimation and rendering for augmented reality
"""

import cv2
import numpy as np
import time
import argparse
import ctypes
from camera_calibration import CameraCalibrator
from pose_estimation import PoseEstimator
from ar_renderer import SimpleARRenderer
from android_camera import get_android_camera


class ARSystem:
    """Main AR System integrating detection, tracking, and rendering"""
    
    def __init__(self, calibration_file='calibration.pkl', 
                 checkerboard_size=(7, 9), square_size=0.020):
        """
        Initialize AR system
        
        Args:
            calibration_file: Path to calibration file
            checkerboard_size: Checkerboard pattern size
            square_size: Square size in meters
        """
        # Load calibration data
        calib_data = CameraCalibrator.load_calibration(calibration_file)
        
        if calib_data is None:
            raise ValueError(f"No calibration data found at {calibration_file}. "
                           "Please run camera_calibration.py first")
        
        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['distortion_coeffs']
        
        # Initialize components
        self.pose_estimator = PoseEstimator(
            self.camera_matrix, self.dist_coeffs,
            checkerboard_size, square_size
        )
        
        self.renderer = SimpleARRenderer(self.camera_matrix, self.dist_coeffs)
        
        # Performance metrics
        self.frame_times = []
        self.max_frame_times = 30
        
        # Rendering mode
        self.render_mode = 'cube'  # 'cube', 'pyramid', 'axes', or 'all'

        # Growth animation state for overlays (0.0..1.0)
        self.grow_value = 0.0
        self.grow_step = 0.12  # amount to increase per processed frame
        
    def process_frame(self, frame):
        """
        Process single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with AR content
        """
        start_time = time.time()
        
        # Detect and estimate pose (pose estimator may skip detection internally)
        pose_data = self.pose_estimator.detect_and_estimate_pose(frame)

        if pose_data:
            rvec = pose_data['rotation_vector']
            tvec = pose_data['translation_vector']

            # Increase growth value up to 1.0 when we have a pose
            self.grow_value = min(1.0, self.grow_value + self.grow_step)

            # Render virtual objects based on mode using 'grow' variants
            if self.render_mode in ['cube', 'all']:
                frame = self.renderer.render_cube_opencv_grow(frame, rvec, tvec, size=0.05, grow=self.grow_value)

            if self.render_mode in ['pyramid', 'all']:
                # Offset pyramid slightly
                tvec_offset = tvec.copy()
                tvec_offset[0] += 0.06
                frame = self.renderer.render_pyramid_opencv_grow(frame, rvec, tvec_offset, size=0.04, grow=self.grow_value)

            if self.render_mode in ['axes', 'all']:
                frame = self.renderer.render_axes_grow(frame, rvec, tvec, length=0.05, grow=self.grow_value)

            # Display pose information
            t = tvec.flatten()
            cv2.putText(frame, f"Position: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})m",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display stability metrics
            stability = self.pose_estimator.get_pose_stability()
            if stability:
                std = stability['translation_std']
                cv2.putText(frame, f"Stability (mm): ({std[0]*1000:.2f}, {std[1]*1000:.2f}, {std[2]*1000:.2f})",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Reset growth when no checkerboard detected so overlays grow anew when detected
            self.grow_value = 0.0
            cv2.putText(frame, "No checkerboard detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate and display FPS
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        avg_frame_time = np.mean(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        latency_ms = avg_frame_time * 1000
        
        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms",
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 0), 2)
        
        # Display current render mode
        cv2.putText(frame, f"Mode: {self.render_mode.upper()} (Press 1-4 to change)",
                   (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0, use_android=False, android_method='ipwebcam', android_kwargs=None, frame_skip=1):
        """
        Run AR system
        
        Args:
            camera_id: Camera device ID (for local camera)
            use_android: Use Android phone camera instead of local camera
            android_method: Method for Android connection ('ipwebcam', 'droidcam', 'rtsp')
            android_kwargs: Additional arguments for Android camera
        """
        if use_android:
            print("=== Connecting to Android Phone Camera ===")
            if android_kwargs is None:
                android_kwargs = {}
            
            cap = get_android_camera(android_method, **android_kwargs)
            
            if not cap.open():
                raise RuntimeError("Could not open Android camera")
            
            print("✓ Android camera connected successfully\n")
        else:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                raise RuntimeError("Could not open camera")
        
        print("=== AR System Running ===")
        print("Camera Source:", "Android Phone" if use_android else f"Local Camera {camera_id}")
        print("\nControls:")
        print("  1 - Cube mode")
        print("  2 - Pyramid mode")
        print("  3 - Axes mode")
        print("  4 - All objects")
        print("  S - Save performance report")
        print("  ESC - Exit")
        print()
        
        frame_idx = 0
        last_processed_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            frame_idx += 1

            # Decide whether to process this frame or reuse last processed result
            do_process = (frame_idx % max(1, frame_skip) == 0)

            if do_process or last_processed_frame is None:
                # Run the full processing pipeline on this frame
                last_processed_frame = self.process_frame(frame)

            # Display (scale to fit screen in portrait mode without altering processing)
            # Create a resizable window once
            try:
                if not hasattr(self, '_display_window_created') or not self._display_window_created:
                    cv2.namedWindow('AR System', cv2.WINDOW_NORMAL)
                    # Get screen size (Windows)
                    try:
                        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
                        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
                    except Exception:
                        # Fallback to typical 1080p
                        screen_w, screen_h = 1920, 1080

                    # Reserve small margin for taskbar and borders
                    self._display_screen_w = screen_w
                    self._display_screen_h = screen_h
                    self._display_margin = 100
                    self._display_window_created = True

                display_frame = last_processed_frame.copy() if last_processed_frame is not None else frame.copy()

                h, w = display_frame.shape[:2]

                # If frame is landscape, rotate it for portrait display (only for showing)
                rotated_for_display = False
                if w > h:
                    display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)
                    rotated_for_display = True
                    h, w = display_frame.shape[:2]

                # Compute scale to fit within screen while preserving aspect ratio
                max_w = max(100, self._display_screen_w - self._display_margin)
                max_h = max(100, self._display_screen_h - self._display_margin)
                scale = min(1.0, float(max_w) / float(w), float(max_h) / float(h))

                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    display_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    cv2.resizeWindow('AR System', new_w, new_h)
                else:
                    cv2.resizeWindow('AR System', w, h)

                cv2.imshow('AR System', display_frame)
            except Exception:
                # If anything goes wrong with display scaling, fall back
                cv2.imshow('AR System', last_processed_frame if last_processed_frame is not None else frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('1'):
                self.render_mode = 'cube'
                print("Render mode: Cube")
            elif key == ord('2'):
                self.render_mode = 'pyramid'
                print("Render mode: Pyramid")
            elif key == ord('3'):
                self.render_mode = 'axes'
                print("Render mode: Axes")
            elif key == ord('4'):
                self.render_mode = 'all'
                print("Render mode: All")
            elif key == ord('s') or key == ord('S'):
                self.save_performance_report()
        
        cap.release()
        cv2.destroyAllWindows()
        
    def save_performance_report(self):
        """Save performance metrics to file"""
        stability = self.pose_estimator.get_pose_stability()
        
        report = "=== AR System Performance Report ===\n\n"
        report += f"Average Frame Time: {np.mean(self.frame_times)*1000:.2f} ms\n"
        report += f"Average FPS: {1.0/np.mean(self.frame_times):.2f}\n"
        report += f"Frame Time Std Dev: {np.std(self.frame_times)*1000:.2f} ms\n"
        report += f"Min Frame Time: {np.min(self.frame_times)*1000:.2f} ms\n"
        report += f"Max Frame Time: {np.max(self.frame_times)*1000:.2f} ms\n\n"
        
        if stability:
            report += "Pose Stability:\n"
            report += f"  Translation Std Dev (mm): {stability['translation_std']*1000}\n"
            report += f"  Rotation Std Dev (rad): {stability['rotation_std']}\n"
            report += f"  Sample Count: {stability['sample_count']}\n"
        
        filename = f"performance_report_{int(time.time())}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"\nPerformance report saved to {filename}")
        print(report)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AR System with Checkerboard Tracking')
    parser.add_argument('--calibration', default='calibration.pkl',
                       help='Path to calibration file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (for local camera)')
    parser.add_argument('--checkerboard-width', type=int, default=7,
                       help='Checkerboard width (inner corners)')
    parser.add_argument('--checkerboard-height', type=int, default=9,
                       help='Checkerboard height (inner corners)')
    parser.add_argument('--square-size', type=float, default=0.020,
                       help='Checkerboard square size in meters')
    
    # Android camera options
    parser.add_argument('--android', dest='android_method', 
                       choices=['ipwebcam'],
                       help='Use Android phone camera (IP Webcam)')
    parser.add_argument('--url', default='http://192.168.1.100:8080',
                       help='IP Webcam URL (for --android ipwebcam)')
    parser.add_argument('--device-id', type=int, default=1,
                       help='DroidCam device ID (for --android droidcam)')
    parser.add_argument('--rtsp-url', default='rtsp://192.168.1.100:8554/live',
                       help='RTSP stream URL (for --android rtsp)')
    parser.add_argument('--android-threaded', action='store_true',
                        help='Use threaded reader for Android camera to reduce latency')
    parser.add_argument('--android-max-fps', type=float, default=0,
                        help='Limit max read FPS for threaded reader (0 = unlimited)')
    parser.add_argument('--android-max-width', type=int, default=0,
                        help='Resize frames to this max width (0 = no resize)')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process only every Nth frame (1 = every frame, 2 = every other frame)')
    
    args = parser.parse_args()
    
    try:
        ar_system = ARSystem(
            calibration_file=args.calibration,
            checkerboard_size=(args.checkerboard_width, args.checkerboard_height),
            square_size=args.square_size
        )
        
        # Prepare Android camera arguments
        use_android = args.android_method is not None
        android_kwargs = {}
        
        if use_android:
            # Only IP Webcam supported
            android_kwargs['url'] = args.url
            android_kwargs['threaded'] = args.android_threaded
            if args.android_max_fps > 0:
                android_kwargs['max_fps'] = args.android_max_fps
            if args.android_max_width > 0:
                android_kwargs['max_width'] = args.android_max_width
        
        ar_system.run(
            camera_id=args.camera,
            use_android=use_android,
            android_method=args.android_method or 'ipwebcam',
            android_kwargs=android_kwargs
            ,
            frame_skip=args.frame_skip
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
