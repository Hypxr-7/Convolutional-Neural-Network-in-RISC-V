import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Hardcode image number here (0-based index)
image_number = 1  # Change this to any valid index < len(train_images)

# Select the image and label
image = train_images[image_number]
label = train_labels[image_number]

# Normalize pixel values to [0, 1] floats
image_normalized = image.astype(np.float32) / 255.0

# Flatten to 28*28 floats (or keep 2D if you want formatting)
flattened_values = image_normalized.flatten()

# Write to file with spaces separating values
output_filename = f"mnist_image_{image_number}.txt"
with open(output_filename, "w") as f:
    for i, val in enumerate(flattened_values):
        f.write(f"{val:.6f} ")
        # Optional: Add newline after every 28 values for readability
        if (i + 1) % 28 == 0:
            f.write("\n")

print(f"Image number {image_number} label: {label}")
print(f"Pixel values saved to {output_filename}")
