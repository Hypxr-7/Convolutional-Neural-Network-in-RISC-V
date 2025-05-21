import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
from tensorflow.keras.datasets import mnist


# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Hardcode image number here (0-based index)
image_number = 33  # Change this to any valid index < len(train_images)

# Select the image and label
image = train_images[image_number]
label = train_labels[image_number]

# Normalize pixel values to [0, 1] floats
image_normalized = image.astype(np.float32) / 255.0

# Flatten to 28*28 floats
flattened_values = image_normalized.flatten()

# Write to file with .float prefix and comma-separated values
output_filename = f"num-{label}.txt"
with open(output_filename, "w") as f:
    for i in range(0, len(flattened_values), 28):
        line_values = flattened_values[i:i+28]
        line_str = ", ".join(f"{val:.6f}" for val in line_values)
        f.write(f".float {line_str}\n")

print(f"Image number {image_number} label: {label}")
print(f"Pixel values saved to {output_filename}")
