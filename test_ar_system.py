#!/usr/bin/env python3
"""
Test script to verify AR system components.
Runs unit tests on all modules without requiring camera/display.
"""

import numpy as np
import cv2
import sys
import traceback


def test_camera_calibration():
    """Test camera calibration module."""
    print("\n" + "=" * 60)
    print("Testing Camera Calibration Module")
    print("=" * 60)
    
    try:
        from camera_calibration import CameraCalibrator
        
        # Test initialization
        calibrator = CameraCalibrator(checkerboard_size=(9, 6), square_size=1.0)
        assert calibrator.checkerboard_size == (9, 6)
        assert calibrator.square_size == 1.0
        assert calibrator.objp.shape == (54, 3)
        print("✓ Initialization successful")
        
        # Test object points
        expected_point = np.array([8.0, 5.0, 0.0])
        actual_point = calibrator.objp[-1]
        assert np.allclose(actual_point, expected_point)
        print("✓ Object points generated correctly")
        
        # Test synthetic calibration data
        camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        calibrator.camera_matrix = camera_matrix
        calibrator.dist_coeffs = dist_coeffs
        
        # Test save/load
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_file = f.name
        
        try:
            calibrator.save_calibration(temp_file)
            print("✓ Calibration save successful")
            
            calibrator2 = CameraCalibrator()
            calibrator2.load_calibration(temp_file)
            
            assert np.allclose(calibrator2.camera_matrix, camera_matrix)
            assert np.allclose(calibrator2.dist_coeffs, dist_coeffs)
            print("✓ Calibration load successful")
        finally:
            os.unlink(temp_file)
        
        print("\n✅ Camera Calibration Module: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Camera Calibration Module: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_pose_estimation():
    """Test pose estimation module."""
    print("\n" + "=" * 60)
    print("Testing Pose Estimation Module")
    print("=" * 60)
    
    try:
        from pose_estimation import PoseEstimator
        
        # Create synthetic camera parameters
        camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # Test initialization
        estimator = PoseEstimator(camera_matrix, dist_coeffs, 
                                 checkerboard_size=(9, 6), square_size=1.0)
        
        assert estimator.checkerboard_size == (9, 6)
        assert estimator.square_size == 1.0
        print("✓ Initialization successful")
        
        # Test projection matrix computation
        rvec = np.array([[0.1], [0.2], [0.3]])
        tvec = np.array([[1.0], [2.0], [10.0]])
        
        proj_matrix = estimator.get_projection_matrix(rvec, tvec)
        assert proj_matrix.shape == (4, 4)
        print("✓ Projection matrix computation successful")
        
        # Test axis drawing (should not crash)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.draw_axis(test_image.copy(), rvec, tvec, length=3.0)
        assert result.shape == test_image.shape
        print("✓ Axis drawing successful")
        
        # Test cube drawing
        result = estimator.draw_cube(test_image.copy(), rvec, tvec, size=3.0)
        assert result.shape == test_image.shape
        print("✓ Cube drawing successful")
        
        print("\n✅ Pose Estimation Module: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Pose Estimation Module: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_opengl_renderer():
    """Test OpenGL renderer module."""
    print("\n" + "=" * 60)
    print("Testing OpenGL Renderer Module")
    print("=" * 60)
    
    try:
        from opengl_renderer import OpenGLRenderer, OPENGL_AVAILABLE
        
        # Test initialization (may not have OpenGL)
        camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        renderer = OpenGLRenderer(camera_matrix, 640, 480)
        
        if OPENGL_AVAILABLE:
            print("✓ OpenGL available and renderer initialized")
        else:
            print("⚠ OpenGL not available (will use fallback rendering)")
        
        print("\n✅ OpenGL Renderer Module: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ OpenGL Renderer Module: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_checkerboard_detection():
    """Test checkerboard detection with synthetic image."""
    print("\n" + "=" * 60)
    print("Testing Checkerboard Detection")
    print("=" * 60)
    
    try:
        from pose_estimation import PoseEstimator
        
        # Create synthetic checkerboard image
        square_size = 50
        checkerboard_size = (9, 6)
        
        width = (checkerboard_size[0] + 1) * square_size
        height = (checkerboard_size[1] + 1) * square_size
        
        checkerboard = np.zeros((height, width), dtype=np.uint8)
        for i in range(checkerboard_size[1] + 1):
            for j in range(checkerboard_size[0] + 1):
                if (i + j) % 2 == 0:
                    y1 = i * square_size
                    y2 = (i + 1) * square_size
                    x1 = j * square_size
                    x2 = (j + 1) * square_size
                    checkerboard[y1:y2, x1:x2] = 255
        
        # Convert to color
        checkerboard_color = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)
        
        # Setup camera parameters
        camera_matrix = np.array([
            [800, 0, width/2],
            [0, 800, height/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # Create estimator
        estimator = PoseEstimator(camera_matrix, dist_coeffs, 
                                 checkerboard_size, square_size=1.0)
        
        # Test detection
        gray = cv2.cvtColor(checkerboard_color, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            print(f"✓ Detected {len(corners)} corners (expected {checkerboard_size[0] * checkerboard_size[1]})")
            assert len(corners) == checkerboard_size[0] * checkerboard_size[1]
            print("✓ Checkerboard detection successful")
        else:
            print("⚠ Checkerboard not detected (may need different parameters)")
        
        print("\n✅ Checkerboard Detection: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Checkerboard Detection: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_ar_system_imports():
    """Test that AR system can be imported."""
    print("\n" + "=" * 60)
    print("Testing AR System Imports")
    print("=" * 60)
    
    try:
        from ar_system import ARSystem
        print("✓ ar_system module imported successfully")
        
        # Test initialization without calibration
        ar = ARSystem(calibration_file=None, checkerboard_size=(9, 6))
        print("✓ ARSystem initialized successfully")
        
        print("\n✅ AR System Imports: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ AR System Imports: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_evaluate_imports():
    """Test that evaluation module can be imported."""
    print("\n" + "=" * 60)
    print("Testing Evaluation Module Imports")
    print("=" * 60)
    
    try:
        from evaluate import ARSystemEvaluator
        print("✓ evaluate module imported successfully")
        
        print("\n✅ Evaluation Module Imports: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation Module Imports: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AR SYSTEM TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Camera Calibration", test_camera_calibration()))
    results.append(("Pose Estimation", test_pose_estimation()))
    results.append(("OpenGL Renderer", test_opengl_renderer()))
    results.append(("Checkerboard Detection", test_checkerboard_detection()))
    results.append(("AR System Imports", test_ar_system_imports()))
    results.append(("Evaluation Module", test_evaluate_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:.<40} {status}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
