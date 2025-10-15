import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import json


# --- 1. Load NER model ---
def load_ner_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    ner = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    return ner


# --- 2. Load Image Classification model ---
def load_cv_model(model_path, class_names_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))  # 10 classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(class_names_path, "r") as f:
        class_names = json.load(f)

    return model, class_names


# --- 3. Preprocessing for images ---
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


# --- 4. Predict animal from image ---
def predict_animal(model, class_names, image_tensor, device):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        return class_names[idx], probs[0][idx].item()


# --- 5. Main pipeline ---
def check_image_text_match(text, image_path, ner_model, cv_model, class_names, device):
    # Extract animal names from text
    ner_results = ner_model(text)
    animals_in_text = [ent["word"].lower() for ent in ner_results if ent["entity_group"] == "ANIMAL"]

    if not animals_in_text:
        print("⚠️ No animal entities found in text.")
        return False

    # Predict animal from image
    image_tensor = preprocess_image(image_path)
    predicted_class, confidence = predict_animal(cv_model, class_names, image_tensor, device)

    print(f"🧩 NER detected: {animals_in_text}")
    print(f"🖼️ Image predicted: {predicted_class} (conf={confidence:.3f})")

    # Compare names (simple substring match)
    for animal in animals_in_text:
        if animal in predicted_class.lower() or predicted_class.lower() in animal:
            return True

    return False


def main():
    # --- Paths ---
    ner_model_path = "ner/animals_ner_temp_5_classes/checkpoint-2210"             # folder with NER weights
    cv_model_path = "cv/resnet18_animals_model/model.pth"  # path to trained ResNet18
    class_names_path = "cv/resnet18_animals_model/class_names.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both models
    ner = load_ner_model(ner_model_path)
    cv_model, class_names = load_cv_model(cv_model_path, class_names_path, device)

    # --- Test examples ---
    examples = [
        ("There is a cow in the picture.", "cv/examples/cow.jpg"),
        ("I see a horse running in the field.", "cv/examples/horse.jpg"),
        ("A little spider is crawling on the wall.", "cv/examples/spider.jpg"),
        ("The cat is playing with a ball.", "cv/examples/sheep.jpg"),  # intentionally wrong
    ]

    for text, image in examples:
        print(f"\nText: {text}")
        result = check_image_text_match(text, image, ner, cv_model, class_names, device)
        print(f"✅ Match: {result}")

if __name__ == "__main__":
    main()