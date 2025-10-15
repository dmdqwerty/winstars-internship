import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from .mnist_classifier_interface import MnistClassifierInterface


class FeedForwardNet(nn.Module):
    """
    Simple feed-forward neural network for MNIST classification.
    """
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # First layer output
        x = F.relu(self.fc_in(x))
        # Hidden layer output
        x = F.relu(self.fc_hidden(x))
        # Last layer output - without activation function
        x = self.fc_out(x)
        return x


class NeuralNetworkMnistClassifier(MnistClassifierInterface):
    """
    Feed-forward NN implementation that follows the MnistClassifierInterface.
    """

    def __init__(
            self,
            input_dim=784,
            hidden_dim=256,
            output_dim=10,
            *,
            batch_size=128,
            lr=1e-3,
            epochs=5,
            **kwargs
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FeedForwardNet(input_dim, hidden_dim, output_dim).to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self, X_train, y_train):
        """
        Train the Feed Forward model.
        """
        print(f"Training Feed-Forward NN on {self.device.upper()}...")

        # Convert numpy arrays to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train(True)

        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.epochs}] | Loss: {total_loss / len(train_loader):.4f}")

        print("Training complete.")


    def predict(self, X_test, y_test=None, verbose=True):
        """
        Predict the labels for the test data.
        Optionally evaluate accuracy if y_test is provided.
        """
        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        if y_test is not None and verbose:
            acc = accuracy_score(y_test, preds)
            print(f"Test Accuracy: {acc:.4f}")

        return preds
