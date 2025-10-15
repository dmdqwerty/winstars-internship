# How to set up the project:
1. Install dependencies: pip install -r requirements.txt
2. Go to: https://drive.google.com/drive/folders/1BNyjxmaPLSXXHwwLLRj5_DPsLW2omNEN?usp=sharing

1. Download folder "checkpoint-2210" and place it in task 2/ner/animals_ner_temp_5_classes
2. Download file model.pth and place it in task 2/cv/resnet18_animals_model
3. Run pipeline.py

### In case you want to check the training process you should either download the animal_ner_synthetic_5_classes.jsonl
### and place it in task 2/ner/data or run task 2/ner/data/dataset_creation.py.
## NER model is based on dslim/distilbert-NER and is fine-tuned on a synthetic dataset which contains many named entities and diverse language constructions. MISC class is added as a bucket for all named entities that are not animals which should reduce hallucinations.

## CV model is a ResNet18 model with an additional trained fully-connected layer.