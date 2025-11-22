#!/usr/bin/env python3
"""
Generate a printable checkerboard pattern for AR calibration
"""

import cv2
import numpy as np
import argparse


# Constants
MM_PER_INCH = 25.4  # Millimeters per inch conversion factor


def generate_checkerboard(width, height, square_size_mm, dpi=300):
    """
    Generate a checkerboard pattern image
    
    Args:
        width: Number of inner corners in width
        height: Number of inner corners in height
        square_size_mm: Size of each square in millimeters
        dpi: Dots per inch for printing
        
    Returns:
        Image containing checkerboard pattern
    """
    # Convert mm to pixels
    pixels_per_mm = dpi / MM_PER_INCH
    square_size_px = int(square_size_mm * pixels_per_mm)
    
    # Total number of squares (one more than inner corners)
    num_squares_width = width + 1
    num_squares_height = height + 1
    
    # Calculate image size
    img_width = num_squares_width * square_size_px
    img_height = num_squares_height * square_size_px
    
    # Add white border
    border_size = square_size_px
    total_width = img_width + 2 * border_size
    total_height = img_height + 2 * border_size
    
    # Create white background
    img = np.ones((total_height, total_width), dtype=np.uint8) * 255
    
    # Draw checkerboard pattern
    for i in range(num_squares_height):
        for j in range(num_squares_width):
            # Alternate between black and white
            if (i + j) % 2 == 0:
                y1 = border_size + i * square_size_px
                y2 = y1 + square_size_px
                x1 = border_size + j * square_size_px
                x2 = x1 + square_size_px
                img[y1:y2, x1:x2] = 0  # Black square
    
    return img


def add_info_text(img, width, height, square_size_mm, dpi):
    """Add information text to the image"""
    # Create a copy for text
    img_with_text = img.copy()
    
    # Add border for text
    border_height = 100
    img_final = np.ones((img.shape[0] + border_height, img.shape[1]), dtype=np.uint8) * 255
    img_final[border_height:, :] = img_with_text
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = 0
    
    text_lines = [
        f"Checkerboard Pattern for Camera Calibration",
        f"Inner Corners: {width} x {height}",
        f"Square Size: {square_size_mm} mm",
        f"DPI: {dpi}"
    ]
    
    y_offset = 25
    for i, line in enumerate(text_lines):
        y_pos = y_offset + i * 25
        cv2.putText(img_final, line, (10, y_pos), font, font_scale, color, thickness)
    
    return img_final


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate checkerboard pattern for AR calibration')
    parser.add_argument('--width', type=int, default=9,
                       help='Number of inner corners in width (default: 9)')
    parser.add_argument('--height', type=int, default=6,
                       help='Number of inner corners in height (default: 6)')
    parser.add_argument('--square-size', type=float, default=25,
                       help='Size of each square in millimeters (default: 25)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for printing (default: 300)')
    parser.add_argument('--output', default='checkerboard.png',
                       help='Output filename (default: checkerboard.png)')
    parser.add_argument('--no-text', action='store_true',
                       help='Do not add information text')
    
    args = parser.parse_args()
    
    print(f"Generating checkerboard pattern...")
    print(f"  Inner corners: {args.width} x {args.height}")
    print(f"  Square size: {args.square_size} mm")
    print(f"  DPI: {args.dpi}")
    
    # Generate checkerboard
    img = generate_checkerboard(args.width, args.height, args.square_size, args.dpi)
    
    # Add text if requested
    if not args.no_text:
        img = add_info_text(img, args.width, args.height, args.square_size, args.dpi)
    
    # Save image
    cv2.imwrite(args.output, img)
    
    print(f"\nCheckerboard pattern saved to: {args.output}")
    print(f"Image size: {img.shape[1]} x {img.shape[0]} pixels")
    print(f"\nPrinting instructions:")
    print(f"  1. Print at {args.dpi} DPI (100% scale, no scaling)")
    print(f"  2. Mount on rigid, flat surface")
    print(f"  3. Ensure pattern is not warped or distorted")
    print(f"  4. Verify square size with ruler ({args.square_size} mm)")
    
    # Display preview only if display is available
    if not args.no_text:
        try:
            print(f"\nDisplaying preview... (press any key to close)")
            # Resize for display
            display_height = 800
            aspect_ratio = img.shape[1] / img.shape[0]
            display_width = int(display_height * aspect_ratio)
            img_display = cv2.resize(img, (display_width, display_height))
            
            cv2.imshow('Checkerboard Preview', img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Could not display preview (headless mode or no display): {e}")


if __name__ == "__main__":
    main()
