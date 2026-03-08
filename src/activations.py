import numpy as np

def sigmoid(Z):
    """
    Sigmoid activation.
    Returns:
        A: output of sigmoid(z), same shape as Z
        cache: returns Z as well, for backprop.
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    ReLU activation
    Returns:
        A: output of relu(z), same shape as Z
        cache: returns Z as well, for back prop.
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backwards(dA, cache):
    """
    Backward propagation for a ReLU unit.
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backwards(dA, cache):
    """
    Backward propagation for a sigmoid unit.
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z)) 
    dZ = dA * s * (1 - s)
    return dZ
