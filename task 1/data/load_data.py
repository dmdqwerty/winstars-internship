import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist(test_size=0.2, random_state=42):
    """
    Load MNIST dataset from OpenML, normalize it, and split into train/test.

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays)
    """
    print("Loading MNIST dataset from OpenML...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Normalize pixel values (0–255) → (0–1)
    X = X / 255.0
    y = y.astype(np.int32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_mnist()