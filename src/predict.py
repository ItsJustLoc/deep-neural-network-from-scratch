import sys
import numpy as np
import matplotlib.pyplot as plt

from model import L_model_forward
from utils import preprocess_image, load_parameters

def predict_image(image_path, classes=None):
    """
    Loads a saved model and predicts whether an image is a cat or non-cat.
    """
    parameters = load_parameters("model_params.pkl")
    image_array, image_vector = preprocess_image(image_path)

    AL, _ = L_model_forward(image_vector, parameters)
    prediction = int(AL[0, 0] > 0.5)

    plt.imshow(image_array)
    plt.axis("off")
    plt.title(f"Prediction: {'cat' if prediction == 1 else 'non-cat'} ({AL[0,0]:.4f})")
    plt.show()

    if classes is not None:
        print(f"Predicted class: {classes[prediction].decode('utf-8')}")
    else:
        print(f"Predicted class: {'cat' if prediction == 1 else 'non-cat'}")

    print(f"Model output probability: {AL[0,0]:.6f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        return

    image_path = sys.argv[1]
    predict_image(image_path)

if __name__ == "__main__":
    main()
