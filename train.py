import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines
import torch

from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM

import os
# os.environ['HF_HOME'] = '/scratch/wej36how/RAG2/trasnformers/cache/'
# os.environ['TRANSFORMERS_CACHE'] = '/scratch/wej36how/RAG2/trasnformers2/cache/'

#from llama import BasicModelRunner
logger = logging.getLogger(__name__)
global_config = None


dataset_path = 'dataset.jsonl'
use_hf = False
model_name = "EleutherAI/pythia-70m"

training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

def tokenize_and_split_data(training_config, tokenizer):
  dataset_path = training_config["datasets"]["path"]
  use_hf = training_config["datasets"]["use_hf"]
  print("tokenize", use_hf, dataset_path)
  if use_hf:
    dataset = datasets.load_dataset(dataset_path)
  else:
    dataset = load_dataset(dataset_path, tokenizer)
  train_dataset = dataset["train"]
  test_dataset = dataset["test"]
  return train_dataset, test_dataset

# Tokenize and split data
def load_dataset(dataset_path, tokenizer):
    random.seed(42)
    finetuning_dataset_loaded = datasets.load_dataset("json", data_files=dataset_path, split="train")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = training_config["model"]["max_length"]
    tokenized_dataset = finetuning_dataset_loaded.map(
        get_tokenize_function(tokenizer, max_length), # returns tokenize_function
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )
    tokenized_dataset = tokenized_dataset.with_format("torch")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
    return split_dataset

# Get function for tokenization, based on config parameters
def get_tokenize_function(tokenizer, _max_length):

  def tokenize_function(examples):
    max_length = _max_length
    tokenizer.pad_token = tokenizer.eos_token

    if "question" in examples and "answer" in examples:
      text = examples["question"][0] + examples["answer"][0]
    elif "input" in examples and "output" in examples:
      text = examples["input"][0] + examples["output"][0]
    else:
      text = examples["text"][0]

    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        max_length
    )

    if tokenized_inputs["input_ids"].shape[1] > max_length:
        logger.warn(
            f"Truncating input from {tokenized_inputs['input_ids'].shape[1]} to {max_length}"
        )

    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
    return tokenized_inputs
  
  return tokenize_function

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

print(train_dataset)
print(test_dataset)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")
base_model.to(device)

# Training configuration
num_train_epochs = 8
max_steps = 3
# trained_model_name = f"lamini_docs_{max_steps}_steps"
trained_model_name = f"/scratch/wej36how/RAG2/lamini_docs_{num_train_epochs}_epochs"
output_dir = trained_model_name

# Training arguments
training_args = TrainingArguments(
    learning_rate=1.0e-5,  # Learning rate
    num_train_epochs=num_train_epochs,  # Number of epochs
    #max_steps=max_steps,  # Overrides num_train_epochs if set
    per_device_train_batch_size=1,  # Training batch size
    per_device_eval_batch_size=1,  # Evaluation batch size
    output_dir=output_dir,  # Directory for checkpoints
    overwrite_output_dir=False,  # Do not overwrite existing directory
    evaluation_strategy="steps",  # Evaluate during training
    save_strategy="steps",  # Save checkpoints during training
    save_steps=10000,  # Save checkpoint every 120 steps
    eval_steps=10000,  # Evaluate every 120 steps
    logging_strategy="steps",  # Log during training
    logging_steps=1,  # Log every step
    warmup_steps=1,  # Number of warmup steps
    gradient_accumulation_steps=4,  # Gradient accumulation
    optim="adafactor",  # Use Adafactor optimizer
    load_best_model_at_end=True,  # Load best model after training
    save_total_limit=1,  # Save only 1 checkpoint
    metric_for_best_model="eval_loss",  # Metric to determine the best model
    greater_is_better=False,  # Lower eval_loss is better
)

# Print model memory footprint
print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")

# Initialize the Hugging Face Trainer
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
training_output = trainer.train()

# Save the final model
save_dir = f"./{output_dir}/final"
trainer.save_model(save_dir)
print("Saved model to:", save_dir)