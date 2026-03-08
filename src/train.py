import numpy as np
from data_loader import load_data, preprocess_data
from model import (
        initialize_parameters_deep, 
        L_model_forward, 
        compute_cost,
        L_model_backwards,
        update_parameters,
        )
from utils import plot_costs, save_parameters

def predict(X, Y, parameters):
    """
    predicts binary labels using learned parameters.
    """

    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)
    print(f"Accuracy: {accuracy:.4f}")
    return predictions

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True):
    """
    Trains an L-layer neural network.
    """
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backwards(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost:.6f}")

    return parameters, costs

def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    train_x, test_x = preprocess_data(train_x_orig, test_x_orig)

    layers_dims = [12288, 20, 7, 5, 1]
    parameters = initialize_parameters_deep(layers_dims)

    parameters, costs = L_layer_model(
            train_x,
            train_y,
            layers_dims,
            num_iterations=2500,
            learning_rate=0.0075,
            print_cost=True,
            )

    print("\n Train set results:")
    predict(train_x, train_y, parameters)

    print("\n Test set results:")
    predict(test_x, test_y, parameters)

    plot_costs(costs, learning_rate=0.0075)

    save_parameters(parameters, "model_params.pkl")
    print("\nSaved trained parameters to model_params.pkl")


if __name__ == "__main__":
    main()
