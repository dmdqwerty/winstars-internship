import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support


def tokenize_and_align_labels(example):
    """
    Tokenize the sentence and align original labels with subword tokens.
    """
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=64
    )

    labels = []
    word_ids = tokenized.word_ids()
    prev_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != prev_word_idx:
            label = example["labels"][word_idx]
            labels.append(label2id[label])
        else:
            label = example["labels"][word_idx]
            if label.startswith("B-"):
                label = "I-" + label[2:]
            labels.append(label2id.get(label, -100))
        prev_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized



def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[pred] for pred, label in zip(prediction, label_row) if label != -100]
        for prediction, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[label] for pred, label in zip(prediction, label_row) if label != -100]
        for prediction, label_row in zip(predictions, labels)
    ]

    all_preds = [p for row in true_predictions for p in row]
    all_labels = [l for row in true_labels for l in row]

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a NER model for animal detection.")
    parser.add_argument("--data_path", type=str, default="data/animal_ner_synthetic_5_classes.jsonl",
                        help="Path to the dataset JSONL file")
    parser.add_argument("--model_name", type=str, default="dslim/distilbert-NER",
                        help="Base transformer model name")
    parser.add_argument("--output_dir", type=str, default="./animals_ner_temp_5_classes",
                        help="Where to save the model")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--max_length", type=int, default=50, help="Max tokenized sentence length")
    args = parser.parse_args()


    dataset = load_dataset("json", data_files={"full": args.data_path})["full"]
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Prepare model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=5, ignore_mismatched_sizes=True)


    label_list = ["O", "B-ANIMAL", "I-ANIMAL", "B-MISC", "I-MISC"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    model.config.label2id = label2id
    model.config.id2label = id2label

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_and_align_labels)
    split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]


    # Training setup
    args_train = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_strategy="steps",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
