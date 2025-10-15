import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from .mnist_classifier_interface import MnistClassifierInterface


class ConvNet(nn.Module):
    """
    Simple Convolutional Neural Network for MNIST classification.
    """
    def __init__(self, output_dim=10):
        super().__init__()
        # Input shape: (batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding='same')   # -> (32, 28, 28)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))                                                # -> (32, 14, 14)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding='same')  # -> (64, 14, 14)
        self.pool2 = nn.MaxPool2d(2, 2)                                                             # -> (64, 7, 7)

        # flatten
        self.flatten = nn.Flatten()                                                                 # -> 64*7*7

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_dim)


    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Conv block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten for fully connected layers
        x = self.flatten(x)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CnnMnistClassifier(MnistClassifierInterface):
    """
    Convolutional NN implementation that follows the MnistClassifierInterface.
    """

    def __init__(
            self,
            output_dim=10,
            *,
            batch_size=128,
            lr=1e-3,
            epochs=5,
            **kwargs
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ConvNet(output_dim).to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self, X_train, y_train):
        """
        Train the Convolutional model.
        """
        print(f"Training Convolutional NN on {self.device.upper()}...")

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
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
        Predict labels for the test data.
        Optionally evaluate accuracy if y_test is provided.
        """
        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        if y_test is not None and verbose:
            acc = accuracy_score(y_test, preds)
            print(f"Test Accuracy: {acc:.4f}")

        return preds