import h5py
import numpy as np


def load_data(train_path="data/train_catvnoncat.h5", test_path="data/test_catvnoncat.h5"):
    """
    Loads the train and test datasets from H5 files.
    """
    with h5py.File(train_path, "r") as train_dataset:
        train_x_orig = np.array(train_dataset["train_set_x"][:])
        train_y = np.array(train_dataset["train_set_y"][:]).reshape(1, -1)

    with h5py.File(test_path, "r") as test_dataset:
        test_x_orig = np.array(test_dataset["test_set_x"][:])
        test_y = np.array(test_dataset["test_set_y"][:]).reshape(1, -1)
        classes = np.array(test_dataset["list_classes"][:])

    return train_x_orig, train_y, test_x_orig, test_y, classes


def preprocess_data(train_x_orig, test_x_orig):
    """
    Flattens and normalizes image datasets.

    Expected output shape:
        train_x: (12288, m_train)
        test_x:  (12288, m_test)
    """
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255.0
    test_x = test_x_flatten / 255.0

    return train_x, test_x
