from models.rf_classifier import RandomForestMnistClassifier
from models.nn_classifier import NeuralNetworkMnistClassifier
from models.cnn_classifier import CnnMnistClassifier


class MnistClassifier:
    """
    Unified interface for MNIST classification using one of three algorithms:
    - Random Forest ('rf')
    - Feed-Forward Neural Network ('nn')
    - Convolutional Neural Network ('cnn')
    """

    def __init__(self, algorithm: str = "rf", **kwargs):
        """
        algorithm : str
            Which model to use ('rf', 'nn', or 'cnn')
        kwargs : dict
            Extra parameters passed to the specific model class.
        """
        algorithm = algorithm.lower()
        if algorithm == "rf":
            self.model = RandomForestMnistClassifier(**kwargs)
        elif algorithm == "nn":
            self.model = NeuralNetworkMnistClassifier(**kwargs)
        elif algorithm == "cnn":
            self.model = CnnMnistClassifier(**kwargs)
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from ['rf', 'nn', 'cnn']."
            )

    def train(self, X_train, y_train):
        """Train the selected model."""
        return self.model.train(X_train, y_train)


    def predict(self, X_test, y_test=None):
        """Predict using the selected model."""
        return self.model.predict(X_test, y_test)