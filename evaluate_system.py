#!/usr/bin/env python3
"""
Evaluation Script for AR System
Tests system under various conditions and generates evaluation report
"""

import cv2
import numpy as np
import time
import json
from camera_calibration import CameraCalibrator
from pose_estimation import PoseEstimator


class ARSystemEvaluator:
    """Evaluates AR system performance"""
    
    def __init__(self, calibration_file='calibration.pkl'):
        """Initialize evaluator"""
        calib_data = CameraCalibrator.load_calibration(calibration_file)
        
        if calib_data is None:
            raise ValueError("No calibration data found")
        
        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['distortion_coeffs']
        self.pose_estimator = PoseEstimator(self.camera_matrix, self.dist_coeffs)
        
        self.metrics = {
            'latency': [],
            'detection_rate': [],
            'pose_stability': [],
            'brightness_levels': [],
            'angles': []
        }
    
    def evaluate_latency(self, num_frames=100):
        """
        Evaluate system latency
        
        Args:
            num_frames: Number of frames to process
        """
        print(f"\n=== Latency Evaluation ({num_frames} frames) ===")
        
        cap = cv2.VideoCapture(0)
        latencies = []
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            
            start_time = time.time()
            pose_data = self.pose_estimator.detect_and_estimate_pose(frame)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{num_frames} frames")
        
        cap.release()
        
        self.metrics['latency'] = latencies
        
        print(f"\nLatency Statistics:")
        print(f"  Mean: {np.mean(latencies):.2f} ms")
        print(f"  Median: {np.median(latencies):.2f} ms")
        print(f"  Std Dev: {np.std(latencies):.2f} ms")
        print(f"  Min: {np.min(latencies):.2f} ms")
        print(f"  Max: {np.max(latencies):.2f} ms")
        print(f"  95th percentile: {np.percentile(latencies, 95):.2f} ms")
    
    def evaluate_detection_rate(self, duration=30):
        """
        Evaluate detection rate over time
        
        Args:
            duration: Duration in seconds
        """
        print(f"\n=== Detection Rate Evaluation ({duration}s) ===")
        print("Move the checkerboard around to test detection robustness")
        
        cap = cv2.VideoCapture(0)
        
        start_time = time.time()
        total_frames = 0
        detected_frames = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            total_frames += 1
            pose_data = self.pose_estimator.detect_and_estimate_pose(frame)
            
            if pose_data:
                detected_frames += 1
                # Draw feedback
                cv2.putText(frame, "DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NOT DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show progress
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            cv2.putText(frame, f"Time remaining: {remaining:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            detection_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0
            cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Detection Rate Evaluation', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        detection_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0
        self.metrics['detection_rate'] = {
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'detection_rate': detection_rate
        }
        
        print(f"\nDetection Rate: {detection_rate:.2f}%")
        print(f"Total frames: {total_frames}")
        print(f"Detected frames: {detected_frames}")
    
    def evaluate_pose_stability(self, duration=10):
        """
        Evaluate pose stability (keep checkerboard still)
        
        Args:
            duration: Duration in seconds
        """
        print(f"\n=== Pose Stability Evaluation ({duration}s) ===")
        print("Keep the checkerboard as still as possible")
        
        cap = cv2.VideoCapture(0)
        
        print("Get checkerboard in view and press SPACE to start")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            pose_data = self.pose_estimator.detect_and_estimate_pose(frame)
            
            if pose_data:
                cv2.putText(frame, "Press SPACE to start stability test", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Position checkerboard in view", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Stability Evaluation', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and pose_data:  # SPACE
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Reset pose history
        self.pose_estimator.pose_history = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            pose_data = self.pose_estimator.detect_and_estimate_pose(frame)
            
            if pose_data:
                cv2.putText(frame, "Keep still...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                stability = self.pose_estimator.get_pose_stability()
                if stability:
                    std = stability['translation_std'] * 1000  # Convert to mm
                    cv2.putText(frame, f"Position StdDev (mm): {std}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Stability Evaluation', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        stability = self.pose_estimator.get_pose_stability()
        
        if stability:
            self.metrics['pose_stability'] = {
                'translation_std_mm': (stability['translation_std'] * 1000).tolist(),
                'rotation_std_rad': stability['rotation_std'].tolist(),
                'sample_count': stability['sample_count']
            }
            
            print(f"\nPose Stability Results:")
            print(f"  Translation Std Dev (mm): {stability['translation_std'] * 1000}")
            print(f"  Rotation Std Dev (rad): {stability['rotation_std']}")
            print(f"  Samples: {stability['sample_count']}")
    
    def save_report(self, filename='evaluation_report.json'):
        """Save evaluation report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nEvaluation report saved to {filename}")
        
        # Also save human-readable version
        txt_filename = filename.replace('.json', '.txt')
        with open(txt_filename, 'w') as f:
            f.write("=== AR System Evaluation Report ===\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            if self.metrics['latency']:
                f.write("Latency:\n")
                f.write(f"  Mean: {np.mean(self.metrics['latency']):.2f} ms\n")
                f.write(f"  Median: {np.median(self.metrics['latency']):.2f} ms\n")
                f.write(f"  Std Dev: {np.std(self.metrics['latency']):.2f} ms\n\n")
            
            if self.metrics['detection_rate']:
                dr = self.metrics['detection_rate']
                f.write("Detection Rate:\n")
                f.write(f"  Rate: {dr['detection_rate']:.2f}%\n")
                f.write(f"  Total Frames: {dr['total_frames']}\n")
                f.write(f"  Detected Frames: {dr['detected_frames']}\n\n")
            
            if self.metrics['pose_stability']:
                ps = self.metrics['pose_stability']
                f.write("Pose Stability:\n")
                f.write(f"  Translation Std Dev (mm): {ps['translation_std_mm']}\n")
                f.write(f"  Rotation Std Dev (rad): {ps['rotation_std_rad']}\n")
        
        print(f"Text report saved to {txt_filename}")


def main():
    """Main evaluation routine"""
    print("=== AR System Evaluator ===")
    print("\nThis will evaluate the AR system performance")
    print("Make sure you have calibrated the camera first")
    
    try:
        evaluator = ARSystemEvaluator()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please run camera_calibration.py first")
        return
    
    print("\nSelect evaluation to run:")
    print("1. Latency")
    print("2. Detection Rate")
    print("3. Pose Stability")
    print("4. All evaluations")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ['1', '4']:
        evaluator.evaluate_latency(num_frames=100)
    
    if choice in ['2', '4']:
        evaluator.evaluate_detection_rate(duration=30)
    
    if choice in ['3', '4']:
        evaluator.evaluate_pose_stability(duration=10)
    
    evaluator.save_report()


if __name__ == "__main__":
    main()
