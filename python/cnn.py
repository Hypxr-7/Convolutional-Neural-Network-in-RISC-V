import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1] as specified in the document
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to add channel dimension (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print("Input shape verification:")
print(f"Training data shape: {x_train.shape}")  # Should be (60000, 28, 28, 1)
print(f"Input image dimensions: {x_train.shape[1:3]}")  # Should be (28, 28)

# Define the CNN model exactly as specified in the document
model = keras.Sequential([
    # 1. Convolutional Layer
    # Filter size: f=5×5, Number of filters: K=8, Stride: s=1, Padding: p=0 (valid)
    # Input: 28×28×1 → Output: 24×24×8
    layers.Conv2D(
        filters=8,                    # K=8 filters
        kernel_size=(5, 5),          # f=5×5 filter size
        strides=1,                   # s=1 stride
        padding='valid',             # p=0 (valid convolution)
        activation='relu',           # ReLU activation as specified
        input_shape=(28, 28, 1),
        name='conv_layer'
    ),
    
    # 2. Max Pooling Layer
    # Pool size: 2×2, Stride: s=2 (as specified in document)
    # Input: 24×24×8 → Output: 12×12×8
    layers.MaxPooling2D(
        pool_size=(2, 2),           # 2×2 max pooling
        strides=2,                  # s=2 stride as per document
        name='maxpool_layer'
    ),
    
    # 3. Flatten for Dense Layer
    # Input: 12×12×8 = 1152 → Output: 1152
    layers.Flatten(name='flatten_layer'),
    
    # 4. Fully Connected (Dense) Layer
    # Input: 1152 → Output: 10
    # Note: Using linear activation here, softmax applied separately as per document
    layers.Dense(
        units=10,
        activation='linear',         # Linear activation before softmax
        name='dense_layer'
    ),
    
    # 5. Softmax Layer
    # Apply softmax to get probabilities
    layers.Softmax(name='softmax_layer')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary to verify architecture
print("\nModel Architecture (as per document specification):")
model.summary()

# Verify the mathematical calculations from the document
print("\n" + "="*60)
print("VERIFICATION OF MATHEMATICAL CALCULATIONS FROM DOCUMENT")
print("="*60)

# Convolutional layer output size calculation
input_size = 28
filter_size = 5
stride = 1
padding = 0
conv_output_size = (input_size - filter_size + 2*padding) // stride + 1
print(f"Conv layer output size: ({conv_output_size} × {conv_output_size} × 8) = ({conv_output_size}, {conv_output_size}, 8)")

# Max pooling output size calculation
pool_size = 2
pool_stride = 2
maxpool_output_size = conv_output_size // pool_stride
print(f"MaxPool output size: ({maxpool_output_size} × {maxpool_output_size} × 8) = ({maxpool_output_size}, {maxpool_output_size}, 8)")

# Flattened size calculation
flattened_size = maxpool_output_size * maxpool_output_size * 8
print(f"Flattened size: {maxpool_output_size} × {maxpool_output_size} × 8 = {flattened_size}")

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Extract weights and biases as specified in the document
print("\n" + "="*60)
print("EXTRACTING WEIGHTS AND BIASES (As per Document Section 6)")
print("="*60)

# Extract Convolutional Layer Parameters
conv_layer = model.get_layer('conv_layer')
W_conv, b_conv = conv_layer.get_weights()

print(f"\nConvolutional Layer Parameters:")
print(f"Filters (W_k) shape: {W_conv.shape}")  # Should be (5, 5, 1, 8)
print(f"Biases (b_k) shape: {b_conv.shape}")   # Should be (8,)

# Extract Fully Connected Layer Parameters (as specified in document)
fc_layer = model.get_layer('dense_layer')  # Get the dense layer (before softmax)
W_fc, b_fc = fc_layer.get_weights()

print(f"\nFully Connected Layer Parameters:")
print(f"W_fc shape: {W_fc.shape}")  # Should be (1152, 10) - note: different from document due to TensorFlow convention
print(f"b_fc shape: {b_fc.shape}")   # Should be (10,)

# Note: TensorFlow uses (input_features, output_features) while the document shows (output_features, input_features)
# The actual mathematical operation is still correct

print("\n" + "-"*40)
print("DETAILED PARAMETER VALUES")
print("-"*40)

# Display convolutional filter values
print(f"\nConvolutional Filters (showing first 2 filters):")
for i in range(min(2, 8)):
    print(f"\nFilter {i+1} (5×5):")
    print(W_conv[:, :, 0, i])
    print(f"Bias for filter {i+1}: {b_conv[i]:.6f}")

# Display dense layer parameters (first few values)
print(f"\nDense Layer Weights (first 10×10 submatrix):")
print(W_fc[:10, :10])
print(f"\nDense Layer Biases:")
print(b_fc)

# Save the model for future use
model.save('trained_mnist_model.h5')
print(f"\nModel saved as 'trained_mnist_model.h5'")

# Function to manually perform inference using extracted weights (as per document section 7)
def manual_inference(image, W_conv, b_conv, W_fc, b_fc):
    """
    Perform manual inference using extracted weights
    Following the mathematical formulation in the document
    """
    # Ensure image is the right shape
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # 1. Convolutional layer (manual convolution)
    conv_output = tf.nn.conv2d(image, W_conv, strides=1, padding='VALID')
    conv_output = tf.nn.bias_add(conv_output, b_conv)
    conv_output = tf.nn.relu(conv_output)
    
    # 2. Max pooling
    pool_output = tf.nn.max_pool2d(conv_output, ksize=2, strides=2, padding='VALID')
    
    # 3. Flatten
    flattened = tf.reshape(pool_output, [1, -1])
    
    # 4. Dense layer (fully connected)
    dense_output = tf.matmul(flattened, W_fc) + b_fc
    
    # 5. Softmax
    probabilities = tf.nn.softmax(dense_output)
    
    return probabilities.numpy()[0]

# Test manual inference
print("\n" + "-"*40)
print("MANUAL INFERENCE TEST")
print("-"*40)

# Test on a single image
test_image = x_test[0]
manual_pred = manual_inference(test_image, W_conv, b_conv, W_fc, b_fc)
model_pred = model.predict(np.expand_dims(test_image, 0), verbose=0)[0]

print(f"Manual inference prediction: {np.argmax(manual_pred)}")
print(f"Model prediction: {np.argmax(model_pred)}")
print(f"True label: {np.argmax(y_test[0])}")
print(f"Difference in probabilities (should be very small): {np.max(np.abs(manual_pred - model_pred)):.10f}")

# Function to save extracted weights (as mentioned in document)
def save_extracted_weights(W_conv, b_conv, W_fc, b_fc, filename='extracted_weights.npz'):
    """Save extracted weights as specified in the document"""
    np.savez(filename,
             W_conv=W_conv,     # Convolutional filters
             b_conv=b_conv,     # Convolutional biases
             W_fc=W_fc,         # Fully connected weights
             b_fc=b_fc)         # Fully connected biases
    print(f"\nExtracted weights saved to {filename}")

# Save the extracted weights
save_extracted_weights(W_conv, b_conv, W_fc, b_fc)

# Function to save all weights and biases to a text file
def save_weights_to_text_file(W_conv, b_conv, W_fc, b_fc, filename='cnn_weights_and_biases.txt'):
    """
    Save all weights and biases to a human-readable text file
    """
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CNN WEIGHTS AND BIASES - MNIST MODEL\n")
        f.write("="*80 + "\n")
        f.write(f"Generated from trained model\n")
        f.write(f"Architecture: Conv(8,5x5,s=1,p=0) -> MaxPool(2x2,s=2) -> Dense(1152->10) -> Softmax\n")
        f.write("="*80 + "\n\n")
        
        # ===== CONVOLUTIONAL LAYER WEIGHTS AND BIASES =====
        f.write("1. CONVOLUTIONAL LAYER PARAMETERS\n")
        f.write("-"*50 + "\n")
        f.write(f"Number of filters: {W_conv.shape[3]}\n")
        f.write(f"Filter size: {W_conv.shape[0]}x{W_conv.shape[1]}\n")
        f.write(f"Weight tensor shape: {W_conv.shape}\n")
        f.write(f"Bias vector shape: {b_conv.shape}\n\n")
        
        # Save each convolutional filter
        for k in range(W_conv.shape[3]):  # For each filter
            f.write(f"Filter {k+1} Weights (5x5):\n")
            for i in range(W_conv.shape[0]):
                for j in range(W_conv.shape[1]):
                    f.write(f"{W_conv[i, j, 0, k]:12.8f} ")
                f.write("\n")
            f.write(f"Filter {k+1} Bias: {b_conv[k]:12.8f}\n")
            f.write("\n")
        
        # ===== DENSE LAYER WEIGHTS AND BIASES =====
        f.write("\n" + "="*60 + "\n")
        f.write("2. DENSE (FULLY CONNECTED) LAYER PARAMETERS\n")
        f.write("-"*50 + "\n")
        f.write(f"Weight matrix shape: {W_fc.shape}\n")
        f.write(f"Bias vector shape: {b_fc.shape}\n")
        f.write(f"Input features: {W_fc.shape[0]}\n")
        f.write(f"Output classes: {W_fc.shape[1]}\n\n")
        
        # Save dense layer biases first (more compact)
        f.write("Dense Layer Biases (10 values):\n")
        for i in range(b_fc.shape[0]):
            f.write(f"b_fc[{i}] = {b_fc[i]:12.8f}\n")
        f.write("\n")
        
        # Save dense layer weights (1152 x 10 matrix)
        f.write("Dense Layer Weight Matrix (1152 x 10):\n")
        f.write("Format: W_fc[input_feature][output_class]\n\n")
        
        # Write weights in a structured format
        # Header for output classes
        f.write("        ")
        for j in range(W_fc.shape[1]):
            f.write(f"   Class_{j:02d}   ")
        f.write("\n")
        f.write("        " + "-"*12*W_fc.shape[1] + "\n")
        
        # Write weights row by row
        for i in range(W_fc.shape[0]):
            f.write(f"F_{i:04d}:")
            for j in range(W_fc.shape[1]):
                f.write(f" {W_fc[i, j]:11.8f}")
            f.write("\n")
            
            # Add spacing every 100 rows for readability
            if (i + 1) % 100 == 0 and i < W_fc.shape[0] - 1:
                f.write("\n")
        
        # ===== SUMMARY STATISTICS =====
        f.write("\n" + "="*60 + "\n")
        f.write("3. SUMMARY STATISTICS\n")
        f.write("-"*30 + "\n")
        
        # Convolutional layer stats
        f.write("Convolutional Layer:\n")
        f.write(f"  Weight range: [{np.min(W_conv):.8f}, {np.max(W_conv):.8f}]\n")
        f.write(f"  Weight mean: {np.mean(W_conv):.8f}\n")
        f.write(f"  Weight std: {np.std(W_conv):.8f}\n")
        f.write(f"  Bias range: [{np.min(b_conv):.8f}, {np.max(b_conv):.8f}]\n")
        f.write(f"  Bias mean: {np.mean(b_conv):.8f}\n")
        f.write(f"  Bias std: {np.std(b_conv):.8f}\n\n")
        
        # Dense layer stats
        f.write("Dense Layer:\n")
        f.write(f"  Weight range: [{np.min(W_fc):.8f}, {np.max(W_fc):.8f}]\n")
        f.write(f"  Weight mean: {np.mean(W_fc):.8f}\n")
        f.write(f"  Weight std: {np.std(W_fc):.8f}\n")
        f.write(f"  Bias range: [{np.min(b_fc):.8f}, {np.max(b_fc):.8f}]\n")
        f.write(f"  Bias mean: {np.mean(b_fc):.8f}\n")
        f.write(f"  Bias std: {np.std(b_fc):.8f}\n\n")
        
        # Total parameters
        total_conv_params = np.prod(W_conv.shape) + np.prod(b_conv.shape)
        total_dense_params = np.prod(W_fc.shape) + np.prod(b_fc.shape)
        total_params = total_conv_params + total_dense_params
        
        f.write("Parameter Count:\n")
        f.write(f"  Convolutional layer: {total_conv_params:,} parameters\n")
        f.write(f"  Dense layer: {total_dense_params:,} parameters\n")
        f.write(f"  Total: {total_params:,} parameters\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF FILE\n")
        f.write("="*60 + "\n")
    
    print(f"\nAll weights and biases saved to '{filename}'")
    print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")

# Import os for file size check
import os

# Save all weights and biases to text file
save_weights_to_text_file(W_conv, b_conv, W_fc, b_fc)

# Also create a compact version with just the numerical values
def save_weights_compact_format(W_conv, b_conv, W_fc, b_fc, filename='cnn_weights_compact.txt'):
    """
    Save weights in a more compact format suitable for importing into other programs
    """
    with open(filename, 'w') as f:
        # Convolutional weights (flattened)
        f.write("# Convolutional Weights (8 filters, 5x5 each)\n")
        f.write("CONV_WEIGHTS = [\n")
        for k in range(W_conv.shape[3]):
            f.write(f"    # Filter {k+1}\n    [")
            filter_weights = W_conv[:, :, 0, k].flatten()
            for i, w in enumerate(filter_weights):
                if i % 5 == 0 and i > 0:
                    f.write("\n     ")
                f.write(f"{w:.8f}")
                if i < len(filter_weights) - 1:
                    f.write(", ")
            f.write("]")
            if k < W_conv.shape[3] - 1:
                f.write(",")
            f.write("\n")
        f.write("]\n\n")
        
        # Convolutional biases
        f.write("# Convolutional Biases (8 values)\n")
        f.write("CONV_BIASES = [")
        for i, b in enumerate(b_conv):
            f.write(f"{b:.8f}")
            if i < len(b_conv) - 1:
                f.write(", ")
        f.write("]\n\n")
        
        # Dense weights
        f.write("# Dense Layer Weights (1152 x 10)\n")
        f.write("DENSE_WEIGHTS = [\n")
        for i in range(W_fc.shape[0]):
            f.write("    [")
            for j in range(W_fc.shape[1]):
                f.write(f"{W_fc[i, j]:.8f}")
                if j < W_fc.shape[1] - 1:
                    f.write(", ")
            f.write("]")
            if i < W_fc.shape[0] - 1:
                f.write(",")
            f.write("\n")
        f.write("]\n\n")
        
        # Dense biases
        f.write("# Dense Layer Biases (10 values)\n")
        f.write("DENSE_BIASES = [")
        for i, b in enumerate(b_fc):
            f.write(f"{b:.8f}")
            if i < len(b_fc) - 1:
                f.write(", ")
        f.write("]\n")
    
    print(f"\nCompact weights format saved to '{filename}'")
    print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")

# Save compact format as well
save_weights_compact_format(W_conv, b_conv, W_fc, b_fc)

# Verification of layer outputs (matching document specifications)
print("\n" + "-"*40)
print("LAYER OUTPUT VERIFICATION")
print("-"*40)

# Test with a single image
test_input = np.expand_dims(x_test[0], 0)

# Conv layer output
conv_out = model.layers[0](test_input)
print(f"Conv layer output shape: {conv_out.shape}")  # Should be (1, 24, 24, 8)

# MaxPool layer output  
maxpool_out = model.layers[1](conv_out)
print(f"MaxPool layer output shape: {maxpool_out.shape}")  # Should be (1, 12, 12, 8)

# Flattened output
flatten_out = model.layers[2](maxpool_out)
print(f"Flatten layer output shape: {flatten_out.shape}")  # Should be (1, 1152)

# Dense layer output (before softmax)
dense_out = model.layers[3](flatten_out)
print(f"Dense layer output shape: {dense_out.shape}")  # Should be (1, 10)

# Final softmax output
softmax_out = model.layers[4](dense_out)
print(f"Softmax layer output shape: {softmax_out.shape}")  # Should be (1, 10)
print(f"Softmax output sum (should be 1.0): {np.sum(softmax_out):.6f}")

# Display the actual equation implementations
print("\n" + "="*60)
print("MATHEMATICAL EQUATIONS IMPLEMENTED")
print("="*60)
print("1. Convolution: O[i,j,k] = Σ Σ X[i+u,j+v] * W_k[u,v] + b_k")
print("2. ReLU: ReLU(x) = max(0, x)")
print("3. Max Pooling: reduces 24×24 to 12×12 with 2×2 windows")
print("4. Dense Layer: Y = W_fc * A' + b_fc")
print("5. Softmax: P_i = exp(Y_i) / Σ exp(Y_j)")
print("="*60)
