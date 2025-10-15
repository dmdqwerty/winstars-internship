from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


if __name__ == '__main__':
    model_path = input("Provide a local model path(such as ./animals_ner_temp_5_classes/checkpoint-2210): ")

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    print(ner_pipeline([
        "A huge hippo yawned lazily in the muddy water, showing off its massive teeth.",
        "That colorful little hummingbird can beat its wings up to 80 times per second while hovering.",
        "The agile cheetah is the fastest land animal, capable of reaching incredible speeds while hunting",
        "Sheep and goats are Lasker's favorite animals",
        "African elephant is the biggest mammal on Earth",
        "There is a white horse in the background.",
        "Did you know a tiny tardigrade can survive the vacuum of space?",
        "The cassowary, known for its bright blue head, is considered the most dangerous bird.",
        "What kind of insect, if any, is a walking stick, and where does it live?",
        "Contrary to popular belief, the domestic ferret is not a rodent but a member of the weasel family.",
        "Upon seeing the bioluminescent glow, the marine biologist identified the organism as a comb jelly."
    ]))