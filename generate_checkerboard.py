#!/usr/bin/env python3
"""
Generate a checkerboard pattern image for printing.
"""

import cv2
import numpy as np
import argparse


def generate_checkerboard(width=9, height=6, square_size=50, output='checkerboard.png'):
    """
    Generate a checkerboard pattern.
    
    Args:
        width: Number of internal corners horizontally
        height: Number of internal corners vertically
        square_size: Size of each square in pixels
        output: Output filename
    """
    # Number of squares is corners + 1
    num_squares_x = width + 1
    num_squares_y = height + 1
    
    # Create checkerboard
    img_width = num_squares_x * square_size
    img_height = num_squares_y * square_size
    
    checkerboard = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for i in range(num_squares_y):
        for j in range(num_squares_x):
            if (i + j) % 2 == 0:
                y1 = i * square_size
                y2 = (i + 1) * square_size
                x1 = j * square_size
                x2 = (j + 1) * square_size
                checkerboard[y1:y2, x1:x2] = 255
    
    # Add white border
    border_size = square_size
    bordered = cv2.copyMakeBorder(
        checkerboard, 
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, 
        value=255
    )
    
    # Save image
    cv2.imwrite(output, bordered)
    
    print(f"Checkerboard generated: {output}")
    print(f"Pattern size: {num_squares_x}x{num_squares_y} squares ({width}x{height} internal corners)")
    print(f"Image size: {bordered.shape[1]}x{bordered.shape[0]} pixels")
    print(f"Square size: {square_size} pixels")
    print(f"\nTo use with AR system:")
    print(f"  python ar_system.py --calibrate --checkerboard-width {width} --checkerboard-height {height}")


def main():
    parser = argparse.ArgumentParser(description='Generate checkerboard pattern for AR calibration')
    parser.add_argument('--width', type=int, default=9,
                       help='Internal corners width (default: 9)')
    parser.add_argument('--height', type=int, default=6,
                       help='Internal corners height (default: 6)')
    parser.add_argument('--square-size', type=int, default=50,
                       help='Square size in pixels (default: 50)')
    parser.add_argument('--output', type=str, default='checkerboard.png',
                       help='Output filename (default: checkerboard.png)')
    
    args = parser.parse_args()
    
    generate_checkerboard(
        width=args.width,
        height=args.height,
        square_size=args.square_size,
        output=args.output
    )


if __name__ == '__main__':
    main()
