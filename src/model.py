import numpy as np
from activations import relu, sigmoid, relu_backwards, sigmoid_backwards

def initialize_parameters_deep(layer_dims):
    """
    Initializes parameters for an L-layer Neural Network.

    Args:
    layer_dims -- list containing dimensions of each layer

    Returns:
    parameters -- dictionary containing:
                    W1, b1, ..., WL, bL
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f"W{l}"] = (
                    np.random.randn(layer_dims[l], layer_dims[l - 1])
                    / np.sqrt(layer_dims[l - 1])
                )
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

        assert parameters[f"W{l}"].shape == (layer_dims[l], layer_dims[l-1])
        assert parameters[f"b{l}"].shape == (layer_dims[l], 1)

    return parameters

def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Args:
    A -- activations from previous layer (or input data), shape (size_prev, m)
    W -- weights matrix, shape (size_curr, size_prev)
    b -- bias vector, shape (size_curr, 1)

    Returns:
    Z -- pre-activation parameter
    cache -- tuple containing (A, W, b)
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    assert Z.shape == (W.shape[0], A.shape[1])
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implements forward propagation for one layer:
    LINEAR -> ACTIVATION

    Args:
    A_prev -- activations from previous layer
    W -- weights matrix
    b -- bias vector
    activation -- "relu" or "sigmoid"

    Returns:
    A -- output of the activation
    cache -- tuple of (linear_cache, activation_cache)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    else:
        raise ValueError("activation must be 'relu' or 'sigmoid'")

    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    """
    Implements forward propagation for:
    [LINEAR -> RELU] * (L-1) -> LINEAR -> SIGMOID

    Args:
    X -- data, shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches for backprop
    """
    caches = []
    A = X
    L = len(parameters) // 2

    # Hidden layers: ReLU
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
                A_prev,
                parameters[f"W{l}"],
                parameters[f"b{l}"],
                activation="relu"
                )
        caches.append(cache)

    # Output layer: Sigmoid
    AL, cache = linear_activation_forward(
            A,
            parameters[f"W{L}"],
            parameters[f"b{L}"],
            activation="sigmoid"
            )
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])
    return AL, caches

def compute_cost(AL, Y):
    """
    Implements binary cross-entropy cost.

    Args:
    AL -- predicted probabilities, shape (1, m)
    Y -- true labels, shape (1, m)

    Returns:
    cost -- scalar cost
    """
    m = Y.shape[1]

    cost = -(1 / m) * np.sum(
            Y * np.log(AL) + (1 - Y) * np.log(1 - AL)
            )

    cost = np.squeeze(cost)
    assert cost.shape == ()

    return cost

def linear_backwards(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer.

    Args:
    dZ -- Gradient of the cost with respect to the linear output Z
    cache -- tuple of values (A_prev, W, b) from forward propagation
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implements backward propagation for one layer:
    LINEAR -> ACTIVATION

    Args:
    dA -- post-activation gradient for current layer
    cache -- tuple of (linear_cache, activation_cache)
    activation -- "relu" or "sigmoid"

    Returns:
    dA_prev -- gradient with respect to previous layer activation
    dW -- gradient with respect to current layer weights
    db -- gradient with respent to current layer bias
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backwards(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backwards(dA, activation_cache)
    else:
        raise ValueError("activation must be 'relu' or 'sigmoid'")

    dA_prev, dW, db = linear_backwards(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backwards(AL, Y, caches):
    """
    Implements backward propagation for the full network:
    [LINEAR -> RELU] * (L-1) -> LINEAR -> SIGMOID

    Args:
    AL -- probability vector from forward propagation, shape (1, m)
    Y -- true labels, shape (1, m)
    caches -- list of caches from forward propagation

    Returns:
    grads -- dictionary with gradients
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # Derivative of cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Last Layer: sigmoid
    current_cache = caches[L - 1]
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads[f"dA{L - 1}"] = dA_prev
    grads[f"dW{L}"] = dW 
    grads[f"db{L}"] = db

    # Hidden Layers: ReLU

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(
                grads[f"dA{l+1}"], 
                current_cache, 
                activation="relu"
                )
        grads[f"dA{l}"] = dA_prev
        grads[f"dW{l+1}"] = dW 
        grads[f"db{l+1}"] = db


    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent.

    Args:
    parameters -- dictionary containing parameters
    grads -- dictionary containing gradients
    learning_rate -- learning rate for gradient descent

    Returns:
    parameters -- updated parameters
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters[f"W{l}"] = parameters[f"W{l}"] - learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] = parameters[f"b{l}"] - learning_rate * grads[f"db{l}"]

    return parameters









