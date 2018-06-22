import numpy as np

def sigmoid(z, threshold=-np.inf):
    threshold_arr = np.full(np.shape(z), threshold)
    z = np.maximum(threshold_arr, z)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z, threshold=-np.inf):
    return np.multiply(sigmoid(z, threshold), (1.0 - sigmoid(z, threshold)))

def relu(x):
    return np.maximum(x, 0)

def relu_gradient(x):
    return (x >= 0) * 1
