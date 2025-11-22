#!/usr/bin/env python3
"""
Unit tests for AR system components
Tests functionality that doesn't require camera/display
"""

import unittest
import numpy as np
import cv2
import os
import sys
import tempfile
from pathlib import Path


class TestCameraCalibration(unittest.TestCase):
    """Test camera calibration module"""
    
    def test_calibrator_initialization(self):
        """Test CameraCalibrator initialization"""
        from camera_calibration import CameraCalibrator
        
        calibrator = CameraCalibrator(checkerboard_size=(9, 6), square_size=0.025)
        self.assertEqual(calibrator.checkerboard_size, (9, 6))
        self.assertEqual(calibrator.square_size, 0.025)
        self.assertEqual(calibrator.objp.shape, (54, 3))
    
    def test_calibration_save_load(self):
        """Test saving and loading calibration data"""
        from camera_calibration import CameraCalibrator
        
        # Create fake calibration data
        calib_data = {
            'camera_matrix': np.eye(3),
            'distortion_coeffs': np.zeros((5, 1)),
            'rms_error': 0.5
        }
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            test_file = f.name
        
        try:
            # Save
            calibrator = CameraCalibrator()
            calibrator.save_calibration(calib_data, test_file)
            
            # Load
            loaded_data = CameraCalibrator.load_calibration(test_file)
            
            self.assertIsNotNone(loaded_data)
            self.assertTrue(np.allclose(loaded_data['camera_matrix'], calib_data['camera_matrix']))
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)


class TestPoseEstimation(unittest.TestCase):
    """Test pose estimation module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    def test_pose_estimator_initialization(self):
        """Test PoseEstimator initialization"""
        from pose_estimation import PoseEstimator
        
        estimator = PoseEstimator(
            self.camera_matrix, 
            self.dist_coeffs,
            checkerboard_size=(9, 6),
            square_size=0.025
        )
        
        self.assertEqual(estimator.checkerboard_size, (9, 6))
        self.assertEqual(estimator.square_size, 0.025)
        self.assertTrue(np.allclose(estimator.camera_matrix, self.camera_matrix))
    
    def test_pose_stability_empty(self):
        """Test pose stability with no history"""
        from pose_estimation import PoseEstimator
        
        estimator = PoseEstimator(self.camera_matrix, self.dist_coeffs)
        stability = estimator.get_pose_stability()
        
        self.assertIsNone(stability)
    
    def test_pose_history_tracking(self):
        """Test pose history tracking"""
        from pose_estimation import PoseEstimator
        
        estimator = PoseEstimator(self.camera_matrix, self.dist_coeffs)
        
        # Add fake poses
        for i in range(5):
            estimator.pose_history.append({
                'rvec': np.random.randn(3, 1),
                'tvec': np.random.randn(3, 1),
                'timestamp': i
            })
        
        stability = estimator.get_pose_stability()
        
        self.assertIsNotNone(stability)
        self.assertEqual(stability['sample_count'], 5)
        self.assertEqual(len(stability['translation_variance']), 3)
        self.assertEqual(len(stability['rotation_variance']), 3)


class TestARRenderer(unittest.TestCase):
    """Test AR renderer module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    def test_simple_renderer_initialization(self):
        """Test SimpleARRenderer initialization"""
        from ar_renderer import SimpleARRenderer
        
        renderer = SimpleARRenderer(self.camera_matrix, self.dist_coeffs)
        
        self.assertTrue(np.allclose(renderer.camera_matrix, self.camera_matrix))
    
    def test_render_cube(self):
        """Test cube rendering"""
        from ar_renderer import SimpleARRenderer
        
        renderer = SimpleARRenderer(self.camera_matrix, self.dist_coeffs)
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create test pose
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        tvec = np.array([[0.0], [0.0], [0.5]], dtype=np.float32)
        
        # Render cube
        result = renderer.render_cube_opencv(frame, rvec, tvec, size=0.05)
        
        self.assertEqual(result.shape, frame.shape)
        # Check that rendering modified the frame
        self.assertFalse(np.array_equal(result, frame))
    
    def test_render_pyramid(self):
        """Test pyramid rendering"""
        from ar_renderer import SimpleARRenderer
        
        renderer = SimpleARRenderer(self.camera_matrix, self.dist_coeffs)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        tvec = np.array([[0.0], [0.0], [0.5]], dtype=np.float32)
        
        result = renderer.render_pyramid_opencv(frame, rvec, tvec, size=0.05)
        
        self.assertEqual(result.shape, frame.shape)
    
    def test_render_axes(self):
        """Test axes rendering"""
        from ar_renderer import SimpleARRenderer
        
        renderer = SimpleARRenderer(self.camera_matrix, self.dist_coeffs)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        tvec = np.array([[0.0], [0.0], [0.5]], dtype=np.float32)
        
        result = renderer.render_axes(frame, rvec, tvec, length=0.05)
        
        self.assertEqual(result.shape, frame.shape)


class TestCheckerboardGenerator(unittest.TestCase):
    """Test checkerboard generator"""
    
    def test_generate_checkerboard(self):
        """Test checkerboard generation"""
        from generate_checkerboard import generate_checkerboard
        
        img = generate_checkerboard(width=9, height=6, square_size_mm=25, dpi=300)
        
        self.assertIsNotNone(img)
        self.assertEqual(len(img.shape), 2)  # Grayscale image
        self.assertTrue(img.shape[0] > 0)
        self.assertTrue(img.shape[1] > 0)
    
    def test_checkerboard_pattern(self):
        """Test that checkerboard has correct pattern"""
        from generate_checkerboard import generate_checkerboard
        
        img = generate_checkerboard(width=3, height=3, square_size_mm=10, dpi=100)
        
        # Check that image contains both black and white pixels
        self.assertTrue(np.any(img == 0))   # Black pixels
        self.assertTrue(np.any(img == 255)) # White pixels


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the AR system"""
    
    def test_full_calibration_workflow(self):
        """Test calibration save/load workflow"""
        from camera_calibration import CameraCalibrator
        
        # Create calibration data
        calib_data = {
            'camera_matrix': np.random.randn(3, 3),
            'distortion_coeffs': np.random.randn(5, 1),
            'rms_error': 0.5
        }
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            test_file = f.name
        
        try:
            calibrator = CameraCalibrator()
            calibrator.save_calibration(calib_data, test_file)
            
            # Load and verify
            loaded = CameraCalibrator.load_calibration(test_file)
            self.assertTrue(np.allclose(loaded['camera_matrix'], calib_data['camera_matrix']))
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_pose_estimation_with_renderer(self):
        """Test pose estimation and rendering together"""
        from pose_estimation import PoseEstimator
        from ar_renderer import SimpleARRenderer
        
        camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        estimator = PoseEstimator(camera_matrix, dist_coeffs)
        renderer = SimpleARRenderer(camera_matrix, dist_coeffs)
        
        # Create test data
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.array([[0.0], [0.0], [0.5]], dtype=np.float32)
        
        # Render
        result = renderer.render_cube_opencv(frame, rvec, tvec)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCameraCalibration))
    suite.addTests(loader.loadTestsFromTestCase(TestPoseEstimation))
    suite.addTests(loader.loadTestsFromTestCase(TestARRenderer))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckerboardGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
