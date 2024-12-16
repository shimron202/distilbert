# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:13:47 2024

@author: Shimron-Ifrrah
"""

import os
import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    logging
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import TrainerCallback
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_info()  # Set Hugging Face transformers logging level to INFO

# Check GPU availability
print("Checking GPU availability...")
print("Is GPU available?:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

if not torch.cuda.is_available():
    raise RuntimeError("GPU is not available! Please ensure your RTX 1080 is properly configured.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure wandb is disabled
os.environ["WANDB_MODE"] = "disabled"

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
print("Model and tokenizer loaded successfully!")

# Load IMDb dataset
print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Tokenize dataset
print("Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("Dataset tokenized!")

# Define training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,  # Log every 10 steps
    report_to="none",  # Log only to the console
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision
    save_strategy="epoch",  # Save checkpoints after each epoch
)

# Define a custom metric function for Trainer
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

class CustomCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started!")
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        print(f"Step {state.global_step}/{state.max_steps} completed. Time elapsed: {elapsed_time:.2f}s")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch:.2f} completed. Steps completed: {state.global_step}/{state.max_steps}")
        
# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[CustomCallback()]  # Add custom callback
)

# Train the model with progress monitoring
print("Starting training...")
trainer.train()
print("Training complete!")

# Save the model and tokenizer
print("Saving the trained model and tokenizer...")
model.save_pretrained("my_distilbert_model")
tokenizer.save_pretrained("my_distilbert_model")
print("Model and tokenizer saved!")

# Final status
print("Process completed successfully!")
