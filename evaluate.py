#!/usr/bin/env python3
"""
Evaluation script for AR system.
Measures latency, pose stability, and visual alignment.
"""

import cv2
import numpy as np
import argparse
import time
import json
from collections import defaultdict

from camera_calibration import CameraCalibrator
from pose_estimation import PoseEstimator


class ARSystemEvaluator:
    """Evaluates AR system performance."""
    
    def __init__(self, calibration_file, checkerboard_size=(9, 6), square_size=1.0):
        """
        Initialize evaluator.
        
        Args:
            calibration_file (str): Path to camera calibration file
            checkerboard_size (tuple): Tuple of (width, height) internal corners
            square_size (float): Size of checkerboard square
        """
        self.calibrator = CameraCalibrator(checkerboard_size, square_size)
        self.calibrator.load_calibration(calibration_file)
        
        self.pose_estimator = PoseEstimator(
            self.calibrator.camera_matrix,
            self.calibrator.dist_coeffs,
            checkerboard_size,
            square_size
        )
        
        self.metrics = {
            'latency': [],
            'detection_success': [],
            'pose_position': [],
            'pose_rotation': [],
            'timestamps': [],
            'lighting_conditions': [],
            'viewing_angles': []
        }
    
    def evaluate_latency(self, video_source=0, duration=30):
        """
        Evaluate system latency.
        
        Args:
            video_source: Video source
            duration: Evaluation duration in seconds
        """
        print("Evaluating system latency...")
        print(f"Recording for {duration} seconds")
        
        cap = cv2.VideoCapture(video_source)
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Measure processing time
            proc_start = time.time()
            success, rvec, tvec, corners = self.pose_estimator.detect_and_estimate_pose(frame)
            proc_end = time.time()
            
            latency_ms = (proc_end - proc_start) * 1000
            self.metrics['latency'].append(latency_ms)
            self.metrics['detection_success'].append(1 if success else 0)
            self.metrics['timestamps'].append(time.time())
            
            if success:
                self.metrics['pose_position'].append(tvec.flatten().tolist())
                self.metrics['pose_rotation'].append(rvec.flatten().tolist())
            
            frame_count += 1
            
            # Visual feedback
            if success:
                frame = self.pose_estimator.draw_axis(frame, rvec, tvec)
            
            cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Latency Evaluation', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Report results
        self._report_latency_results()
    
    def _report_latency_results(self):
        """Report latency evaluation results."""
        print("\n" + "=" * 60)
        print("LATENCY EVALUATION RESULTS")
        print("=" * 60)
        
        if len(self.metrics['latency']) == 0:
            print("No data collected")
            return
        
        latencies = np.array(self.metrics['latency'])
        detection_rate = np.mean(self.metrics['detection_success']) * 100
        
        print(f"\nTotal frames processed: {len(latencies)}")
        print(f"Detection success rate: {detection_rate:.1f}%")
        print(f"\nLatency Statistics:")
        print(f"  Mean: {np.mean(latencies):.2f} ms")
        print(f"  Median: {np.median(latencies):.2f} ms")
        print(f"  Std Dev: {np.std(latencies):.2f} ms")
        print(f"  Min: {np.min(latencies):.2f} ms")
        print(f"  Max: {np.max(latencies):.2f} ms")
        print(f"  95th percentile: {np.percentile(latencies, 95):.2f} ms")
        
        fps = 1000.0 / np.mean(latencies)
        print(f"\nEstimated FPS: {fps:.1f}")
        
        # Latency classification
        if np.mean(latencies) < 16.67:
            print("\nPerformance: EXCELLENT (>60 FPS capable)")
        elif np.mean(latencies) < 33.33:
            print("\nPerformance: GOOD (30-60 FPS)")
        elif np.mean(latencies) < 50:
            print("\nPerformance: ACCEPTABLE (20-30 FPS)")
        else:
            print("\nPerformance: POOR (<20 FPS)")
    
    def evaluate_pose_stability(self, video_source=0, duration=30):
        """
        Evaluate pose estimation stability.
        
        Args:
            video_source: Video source
            duration: Evaluation duration in seconds
        """
        print("\nEvaluating pose stability...")
        print(f"Recording for {duration} seconds")
        print("Keep the checkerboard as still as possible")
        
        cap = cv2.VideoCapture(video_source)
        
        start_time = time.time()
        pose_data = []
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            success, rvec, tvec, corners = self.pose_estimator.detect_and_estimate_pose(frame)
            
            if success:
                pose_data.append({
                    'position': tvec.flatten(),
                    'rotation': rvec.flatten(),
                    'timestamp': time.time()
                })
                
                frame = self.pose_estimator.draw_axis(frame, rvec, tvec)
            
            cv2.putText(frame, f"Poses recorded: {len(pose_data)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Keep board still!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Stability Evaluation', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Analyze stability
        self._report_stability_results(pose_data)
    
    def _report_stability_results(self, pose_data):
        """Report stability evaluation results."""
        print("\n" + "=" * 60)
        print("POSE STABILITY EVALUATION RESULTS")
        print("=" * 60)
        
        if len(pose_data) < 2:
            print("Insufficient data collected")
            return
        
        positions = np.array([p['position'] for p in pose_data])
        rotations = np.array([p['rotation'] for p in pose_data])
        
        # Position statistics
        pos_mean = np.mean(positions, axis=0)
        pos_std = np.std(positions, axis=0)
        pos_total_std = np.linalg.norm(pos_std)
        
        print(f"\nTotal pose samples: {len(pose_data)}")
        print(f"\nPosition Statistics:")
        print(f"  Mean: [{pos_mean[0]:.4f}, {pos_mean[1]:.4f}, {pos_mean[2]:.4f}]")
        print(f"  Std Dev: [{pos_std[0]:.4f}, {pos_std[1]:.4f}, {pos_std[2]:.4f}]")
        print(f"  Total Std Dev: {pos_total_std:.4f} units")
        
        # Rotation statistics
        rot_std = np.std(rotations, axis=0)
        rot_total_std = np.linalg.norm(rot_std)
        
        print(f"\nRotation Statistics:")
        print(f"  Std Dev: [{rot_std[0]:.4f}, {rot_std[1]:.4f}, {rot_std[2]:.4f}] rad")
        print(f"  Total Std Dev: {rot_total_std:.4f} rad ({np.degrees(rot_total_std):.2f}°)")
        
        # Jitter analysis (frame-to-frame variation)
        pos_diffs = np.diff(positions, axis=0)
        pos_jitter = np.linalg.norm(pos_diffs, axis=1)
        
        print(f"\nPosition Jitter (frame-to-frame):")
        print(f"  Mean: {np.mean(pos_jitter):.4f} units/frame")
        print(f"  Max: {np.max(pos_jitter):.4f} units/frame")
        
        rot_diffs = np.diff(rotations, axis=0)
        rot_jitter = np.linalg.norm(rot_diffs, axis=1)
        
        print(f"\nRotation Jitter (frame-to-frame):")
        print(f"  Mean: {np.mean(rot_jitter):.4f} rad/frame ({np.degrees(np.mean(rot_jitter)):.2f}°/frame)")
        print(f"  Max: {np.max(rot_jitter):.4f} rad/frame ({np.degrees(np.max(rot_jitter)):.2f}°/frame)")
        
        # Stability rating
        if pos_total_std < 0.01 and rot_total_std < 0.01:
            print("\nStability: EXCELLENT (Very stable)")
        elif pos_total_std < 0.05 and rot_total_std < 0.05:
            print("\nStability: GOOD (Stable)")
        elif pos_total_std < 0.1 and rot_total_std < 0.1:
            print("\nStability: ACCEPTABLE (Some jitter)")
        else:
            print("\nStability: POOR (Unstable)")
    
    def evaluate_viewing_angles(self, video_source=0):
        """
        Evaluate system at different viewing angles.
        
        Args:
            video_source: Video source
        """
        print("\nEvaluating viewing angles...")
        print("Move the camera around the checkerboard")
        print("Press 'r' to record a sample, ESC to finish")
        
        cap = cv2.VideoCapture(video_source)
        
        angle_samples = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            success, rvec, tvec, corners = self.pose_estimator.detect_and_estimate_pose(frame)
            
            if success:
                frame = self.pose_estimator.draw_axis(frame, rvec, tvec)
                
                # Calculate viewing angle (angle from normal)
                rot_mat, _ = cv2.Rodrigues(rvec)
                normal = rot_mat[:, 2]  # Z-axis in camera frame
                view_angle = np.arccos(np.clip(normal[2], -1.0, 1.0))
                view_angle_deg = np.degrees(view_angle)
                
                cv2.putText(frame, f"Viewing angle: {view_angle_deg:.1f} deg", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Samples: {len(angle_samples)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'r' to record", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Viewing Angle Evaluation', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('r') and success:
                rot_mat, _ = cv2.Rodrigues(rvec)
                normal = rot_mat[:, 2]
                view_angle = np.arccos(np.clip(normal[2], -1.0, 1.0))
                
                angle_samples.append({
                    'angle_rad': view_angle,
                    'angle_deg': np.degrees(view_angle),
                    'position': tvec.flatten().tolist(),
                    'rotation': rvec.flatten().tolist()
                })
                print(f"Sample {len(angle_samples)}: {np.degrees(view_angle):.1f}°")
        
        cap.release()
        cv2.destroyAllWindows()
        
        self._report_viewing_angle_results(angle_samples)
    
    def _report_viewing_angle_results(self, angle_samples):
        """Report viewing angle results."""
        print("\n" + "=" * 60)
        print("VIEWING ANGLE EVALUATION RESULTS")
        print("=" * 60)
        
        if len(angle_samples) == 0:
            print("No samples recorded")
            return
        
        angles = [s['angle_deg'] for s in angle_samples]
        
        print(f"\nTotal samples: {len(angles)}")
        print(f"Viewing angle range: {min(angles):.1f}° - {max(angles):.1f}°")
        print(f"Mean viewing angle: {np.mean(angles):.1f}°")
        print(f"Std dev: {np.std(angles):.1f}°")
        
        # Categorize angles
        frontal = sum(1 for a in angles if a < 30)
        oblique = sum(1 for a in angles if 30 <= a < 60)
        extreme = sum(1 for a in angles if a >= 60)
        
        print(f"\nAngle distribution:")
        print(f"  Frontal (<30°): {frontal} samples")
        print(f"  Oblique (30-60°): {oblique} samples")
        print(f"  Extreme (>60°): {extreme} samples")
        
        if max(angles) > 60:
            print("\nRobustness: EXCELLENT (Works at extreme angles)")
        elif max(angles) > 45:
            print("\nRobustness: GOOD (Works at oblique angles)")
        else:
            print("\nRobustness: ACCEPTABLE (Best at frontal views)")
    
    def save_results(self, filename='evaluation_results.json'):
        """Save evaluation results to file."""
        # Convert numpy arrays to lists for JSON serialization
        results = {
            'latency_ms': {
                'mean': float(np.mean(self.metrics['latency'])) if self.metrics['latency'] else None,
                'std': float(np.std(self.metrics['latency'])) if self.metrics['latency'] else None,
                'min': float(np.min(self.metrics['latency'])) if self.metrics['latency'] else None,
                'max': float(np.max(self.metrics['latency'])) if self.metrics['latency'] else None,
            },
            'detection_rate': float(np.mean(self.metrics['detection_success'])) * 100 if self.metrics['detection_success'] else None,
            'total_frames': len(self.metrics['latency']),
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AR System Evaluation')
    parser.add_argument('--calibration-file', type=str,
                       default='calibration_data/camera_calibration.npz',
                       help='Path to calibration file')
    parser.add_argument('--video-source', type=int, default=0,
                       help='Video source')
    parser.add_argument('--test', type=str, default='all',
                       choices=['latency', 'stability', 'angles', 'all'],
                       help='Test to run')
    parser.add_argument('--duration', type=int, default=30,
                       help='Test duration in seconds')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    evaluator = ARSystemEvaluator(args.calibration_file)
    
    if args.test in ['latency', 'all']:
        evaluator.evaluate_latency(args.video_source, args.duration)
    
    if args.test in ['stability', 'all']:
        evaluator.evaluate_pose_stability(args.video_source, args.duration)
    
    if args.test in ['angles', 'all']:
        evaluator.evaluate_viewing_angles(args.video_source)
    
    evaluator.save_results(args.output)


if __name__ == '__main__':
    main()
