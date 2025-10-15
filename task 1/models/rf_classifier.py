import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .mnist_classifier_interface import MnistClassifierInterface


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST dataset.
    Implements the MnistClassifierInterface.
    """

    def __init__(self, n_estimators=100, max_depth=15, random_state=42, **kwargs):
        """
        Initialize the Random Forest classifier with given parameters.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        """
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test, y_test=None, verbose=True):
        """
        Predict the labels for the test data.
        Optionally evaluate accuracy if y_test is provided.
        """
        preds = self.model.predict(X_test)
        if y_test is not None and verbose:
            acc = accuracy_score(y_test, preds)
            print(f"Test Accuracy: {acc:.4f}")
        return preds
