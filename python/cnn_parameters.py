import tensorflow as tf
import numpy as np
import os

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Add channel dimension
x_train = x_train[..., tf.newaxis]  # (60000, 28, 28, 1)
x_test = x_test[..., tf.newaxis]    # (10000, 28, 28, 1)

# Build the minimal CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, kernel_size=5, strides=1, padding='valid', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)  # No softmax here
])

# Compile and train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Save model
# model.save('trained_mnist_model.h5')

# Extract fully connected layer weights
fc_layer = model.layers[-1]
W_fc, b_fc = fc_layer.get_weights()
print("W_fc shape:", W_fc.shape)  # (1152, 10)
print("b_fc shape:", b_fc.shape)  # (10,)

# Save weights to disk
# np.save("W_fc.npy", W_fc)
# np.save("b_fc.npy", b_fc)
