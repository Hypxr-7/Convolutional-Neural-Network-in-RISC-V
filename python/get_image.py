import tensorflow as tf
import numpy as np

# Load MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Select a single image (e.g., the first one)
image = x_train[0]  # shape: (28, 28)
label = y_train[0]

# Normalize the pixel values to [0.0, 1.0]
image_normalized = image.astype(np.float32) / 255.0

# Convert to list of lists
image_list = image_normalized.tolist()

# Write to file in .float format
with open("mnist_image.txt", "w") as f:
    for row in image_list:
        line = ".float " + " ".join(f"{val:.6f}" for val in row)
        f.write(line + "\n")

print(f"Wrote image of digit '{label}' to mnist_image.txt in .float format.")
