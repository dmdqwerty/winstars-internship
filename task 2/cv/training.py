from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
import numpy as np
import json
import os


def create_model(model, num_freeze_layers, num_out_classes):
    """
    Replace the final classification head of ResNet with a custom one
    and optionally freeze the first N layers of the network.
    """
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, num_out_classes)
    )

    # Freeze first `num_freeze_layers` layers
    for i, layer in enumerate(model.children()):
        if i < num_freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

    return model


def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluate the model on a given dataloader.
    Returns accuracy and average loss.
    """
    model.eval()
    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    return accuracy, np.mean(losses)


def train(model, train_loader, val_loader, loss_fn, optimizer, device, n_epochs=3):
    """
    Main training loop.
    Trains the model for a given number of epochs and prints loss/accuracy.
    """
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        model.train(True)
        batch_losses, batch_accs = [], []

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_batch).float().mean().item()

            batch_losses.append(loss.item())
            batch_accs.append(acc)

            # Show training stats every 50 batches
            if (i + 1) % 50 == 0:
                print(f"Train loss: {np.mean(batch_losses):.4f}, Train acc: {np.mean(batch_accs):.4f}")
                batch_losses, batch_accs = [], []

        # Validation step after each epoch
        val_acc, val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Validation loss: {val_loss:.4f}, Validation acc: {val_acc:.4f}\n")

    return model


if __name__ == '__main__':
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("alessiocorrado99/animals10")
    dataset_path = path + "\\raw-img"

    # Preprocessing pipeline
    resnet_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset and split into train/val/test
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=resnet_transforms)

    original_to_english = {
        "cane": "dog",
        "cavallo": "horse",
        "elefante": "elephant",
        "farfalla": "butterfly",
        "gallina": "chicken",
        "gatto": "cat",
        "mucca": "cow",
        "pecora": "sheep",
        "ragno": "spider",
        "scoiattolo": "squirrel"
    }
    full_dataset.classes = [original_to_english[name] for name in full_dataset.classes]
    full_dataset.class_to_idx = {original_to_english[k]: v for k, v in full_dataset.class_to_idx.items()}

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model (ResNet18 backbone)
    model = create_model(models.resnet18(pretrained=True), num_freeze_layers=9, num_out_classes=len(full_dataset.classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model = train(model, train_loader, val_loader, loss_fn, optimizer, device, n_epochs=5)

    # Evaluate on test set
    test_accuracy, _ = evaluate(model, test_loader, loss_fn, device)
    print('Test accuracy:', test_accuracy)

    # Save model and class mapping
    save_dir = "resnet18_animals_model"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    print(f"✅ Model weights saved to {save_dir}/model.pth")

    with open(os.path.join(save_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(full_dataset.classes, f, indent=2)
    print(f"✅ Class names saved to {save_dir}/class_names.json")




