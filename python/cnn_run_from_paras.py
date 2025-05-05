import numpy as np

# Load parameters
W_fc = np.load("data/W_fc.npy")  # shape: (1152, 10)
b_fc = np.load("data/b_fc.npy")  # shape: (10,)

# Assume 'feature_vector' is a (1152,) vector from intermediate CNN
def softmax(logits):
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / np.sum(exp_vals)

def predict(feature_vector):
    z = np.dot(feature_vector, W_fc) + b_fc  # shape: (10,)
    probs = softmax(z)
    return np.argmax(probs), probs

# Example (random input)
# feature_vector = your_feature_extractor(image)
# label, probs = predict(feature_vector)
