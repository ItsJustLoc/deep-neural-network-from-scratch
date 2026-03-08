import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

def plot_costs(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel("costs")
    plt.xlabel("iterations (per hundreds)")
    plt.title(f"Learning rate = {learning_rate}")

    plt.savefig("assets/training_loss.png")
    plt.show()

def preprocess_image(image_path, num_px=64):
    """
    Loads and preprocesses a custom image for prediciton.

    Returns:
        image_array: original resized image array for display
        image_vector: flattened and normalized image of shape (12288, 1)
    """

    image = Image.open(image_path).convert("RGB")
    image = image.resize((num_px, num_px))
    image_array = np.array(image)

    image_vector = image_array.reshape(1, num_px * num_px * 3).T
    image_vector = image_vector / 255.0

    return image_array, image_vector

def save_parameters(parameters, filepath="model_params.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump(parameters, f)

def load_parameters(filepath="model_params.pkl"):
    with open(filepath, "rb") as f:
        parameters = pickle.load(f)
    return parameters
