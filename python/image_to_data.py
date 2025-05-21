#!/usr/bin/env python3
"""
MNIST Image Converter

This script converts a 28x28 black and white PNG image to a format
similar to the MNIST dataset. The MNIST dataset contains 28x28 grayscale
images where pixel values range from 0 (white) to 255 (black).

The output is a text file with 28 rows, each containing 28 comma-separated values,
normalized between 0 and 1 (divided by 255) and formatted to 6 decimal places:
    0.000000, 0.015686, 0.003922, ...
    0.039216, 0.000000, 0.023529, ...
    0.035294, 0.000000, 0.000000, ...

Usage:
    python mnist_converter.py <input_image_path> <output_file_path>

Example:
    python mnist_converter.py digit.png mnist_digit.txt
"""

import sys
import numpy as np
from PIL import Image


def convert_image_to_mnist_format(image_path, output_path):
    """
    Convert a 28x28 PNG image to MNIST-like format.
    
    Args:
        image_path (str): Path to the input PNG image
        output_path (str): Path to save the output file
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Check if the image is 28x28
        if img.size != (28, 28):
            print(f"Error: Image dimensions are {img.size}, not 28x28")
            return False
        
        # Convert to grayscale if it's not already
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # In MNIST, 0 is white and 255 is black, so we invert if needed
        # Check if the image is already inverted (black background)
        mean_value = np.mean(img_array)
        if mean_value > 127:  # If the image is predominantly white
            img_array = 255 - img_array  # Invert the image
        
        # Flatten the array (MNIST format is a 1D array of 784 values)
        flattened = img_array.flatten()
        
        # Reshape back to 28x28 for saving in grid format
        matrix = flattened.reshape(28, 28)
        
        # Normalize by dividing by 255
        normalized_matrix = matrix / 255.0
        
        # Save as a 28x28 grid with comma-space separated values
        with open(output_path, 'w') as f:
            for row in normalized_matrix:
                # Join with comma and space, format to 6 decimal places
                line ='.float ' + ', '.join(f"{pixel:.6f}" for pixel in row)
                f.write(line + '\n')
        
        print(f"Successfully converted {image_path} to MNIST format and saved to {output_path}")
        
        # Print a sample of values for verification
        print("\nFirst few rows of the saved format:")
        with open(output_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:  # Print just first 3 rows
                    print(line.strip()[:40] + "...")  # Show first part of each row
                else:
                    break
        
        return True
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return False


def display_as_matrix(image_path):
    """
    Load an image and display its pixel values as a 28x28 matrix.
    Useful for verification.
    
    Args:
        image_path (str): Path to the input image file
    """
    try:
        # Open the image and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        if img.size != (28, 28):
            print(f"Warning: Image dimensions are {img.size}, not 28x28")
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Check if the image needs to be inverted for MNIST format
        mean_value = np.mean(img_array)
        if mean_value > 127:
            img_array = 255 - img_array
        
        # Normalize the values by dividing by 255
        normalized_array = img_array / 255.0
        
        print("\nSample of normalized pixel values (first 3 rows, first 5 columns):")
        for i in range(3):  # Print only first 3 rows
            sample_row = [f"{pixel:.6f}" for pixel in normalized_array[i][:5]]
            print(', '.join(sample_row) + ", ...")
        
    except Exception as e:
        print(f"Error displaying image as matrix: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mnist_converter.py <input_image_path> <output_file_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_image_to_mnist_format(input_path, output_path)
    display_as_matrix(input_path)