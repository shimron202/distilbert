# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:44:23 2024

@author: Shimron-Ifrrah
"""

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report

# Correct path to the model
model_path = "my_distilbert_model"

# Load the tokenizer and model from the local folder
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model and tokenizer loaded successfully!")

# Load IMDb test dataset
dataset = load_dataset("imdb")
test_dataset = dataset["test"]

# Tokenize the test dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

print("Tokenizing test dataset...")
tokenized_test = test_dataset.map(tokenize_function, batched=True)
print("Tokenization complete!")

# Create PyTorch tensors for evaluation
def get_predictions(model, tokenized_data):
    predictions = []
    true_labels = []
    for example in tokenized_data:
        inputs = {
            "input_ids": torch.tensor(example["input_ids"]).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(example["attention_mask"]).unsqueeze(0).to(device),
        }
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, axis=1).item()
        predictions.append(predicted_label)
        true_labels.append(example["label"])
    return predictions, true_labels

# Run predictions
print("Running predictions on test data...")
predicted_labels, true_labels = get_predictions(model, tokenized_test)

# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=["Negative", "Positive"])
print("Classification Report:\n", report)
