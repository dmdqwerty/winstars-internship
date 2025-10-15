import torch
from torchvision import models, transforms
from PIL import Image
import json
import os


def load_model(model_path, class_names_path, device):
    """
    Loads the trained ResNet18 model and class names.
    """
    # Load saved class names
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Load the model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, len(class_names))
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names


def predict_image(model, class_names, image_path, device):
    """
    Makes a prediction for a single image.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    predicted_class = class_names[top_idx.item()]
    return predicted_class, top_prob.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    model_path = "resnet18_animals_model/model.pth"           # saved model
    class_names_path = "resnet18_animals_model/class_names.json"         # saved class names
    test_paths = ["examples/clown.jpg", "examples/cow.jpg", "examples/sheep.jpg",
                  "examples/horse.jpg", "examples/spider.jpg"]

    # Load model and classes
    model, class_names = load_model(model_path, class_names_path, device)

    # Predict
    for path in test_paths:
        label, prob = predict_image(model, class_names, path, device)
        print(f"Predicted class: {label} (confidence: {prob:.4f})")
