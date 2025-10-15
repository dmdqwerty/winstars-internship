from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    """
    Abstract interface for MNIST classifiers.
    Each classifier must implement 'train' and 'predict' methods.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the classifier on the training data."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Predict labels for the test data."""
        pass
