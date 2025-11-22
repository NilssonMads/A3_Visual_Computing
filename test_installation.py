#!/usr/bin/env python3
"""
Quick test script to verify AR system installation and dependencies
"""

import sys
import importlib


def check_dependency(module_name, package_name=None):
    """Check if a Python module is available"""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False


def check_opencv_features():
    """Check OpenCV installation and features"""
    try:
        import cv2
        print(f"\nOpenCV version: {cv2.__version__}")
        
        # Check for camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera device accessible")
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print("✗ Cannot access camera device")
            
        return True
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        return False


def check_opengl():
    """Check OpenGL installation"""
    try:
        import OpenGL.GL
        import OpenGL.GLU
        print("✓ OpenGL is installed")
        
        import pygame
        print(f"✓ Pygame is installed (version: {pygame.version.ver})")
        return True
    except Exception as e:
        print(f"✗ OpenGL/Pygame error: {e}")
        return False


def main():
    """Main test routine"""
    print("=== AR System Installation Test ===\n")
    
    print("Checking Python version...")
    print(f"Python {sys.version}")
    
    if sys.version_info < (3, 8):
        print("⚠ Warning: Python 3.8 or higher is recommended")
    else:
        print("✓ Python version is compatible\n")
    
    print("Checking dependencies...")
    
    all_ok = True
    
    # Check core dependencies
    all_ok &= check_dependency('cv2', 'opencv-python')
    all_ok &= check_dependency('numpy')
    all_ok &= check_dependency('OpenGL', 'PyOpenGL')
    all_ok &= check_dependency('pygame')
    
    # Check OpenCV features
    print("\nChecking OpenCV features...")
    all_ok &= check_opencv_features()
    
    # Check OpenGL
    print("\nChecking OpenGL...")
    all_ok &= check_opengl()
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("✓ All checks passed!")
        print("\nYou can now proceed with:")
        print("  1. python generate_checkerboard.py")
        print("  2. python camera_calibration.py")
        print("  3. python ar_system.py")
    else:
        print("✗ Some checks failed!")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
