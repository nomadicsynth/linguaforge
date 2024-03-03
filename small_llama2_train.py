import json
import os
import time
import torch
from CustomLlamaModel import CustomLlamaModel
from datasets import load_dataset, Dataset
from transformers import LlamaConfig, AutoTokenizer, TrainingArguments, Trainer


# Model settings
hidden_layers = 12  # Number of transformer layers
hidden_size = 1024  # Size of the hidden states in the transformer layers
intermediate_size = 2048  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
context_length = 2048  # Length of the input context
stride = 50  # Stride for splitting the input into multiple sequences

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "D:/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = 100  # Number of examples to use from the dataset. 0 means all examples
dataset_split = 0.9  # Percentage of examples to use for training

# Training settings
seed = 42
epochs = 5  # Number of training epochs
batch_size = 3  # Number of sequences to process in parallel
gradient_accumulation_steps = 4  # Number of update steps to accumulate before performing a backward pass
# warmup_steps = 100 / gradient_accumulation_steps  # Number of warmup steps for the learning rate scheduler
warmup_steps = 10  # Number of warmup steps for the learning rate scheduler

run = "2"
output_dir = "./results/run-" + run
logging_dir = output_dir + "/logs"
# final_dir = "./final"

learning_rate = 5e-5
lr_scheduler_type = "linear"
optim = "adamw_torch"  # Use PyTorch's AdamW optimizer

logging_steps = 10  # Log training loss every X steps

evaluation_strategy = "steps"
eval_steps = 1 / (epochs * 4)  # Evaluate every 25% of an epoch
save_strategy = "steps"
save_steps = eval_steps

load_best_model_at_end = True
metric_for_best_model = "loss"

# Write the configuration to a JSON file
training_config = {
    "hidden_layers": hidden_layers,
    "hidden_size": hidden_size,
    "intermediate_size": intermediate_size,
    "attention_heads": attention_heads,
    "context_length": context_length,
    "stride": stride,
    "seed": seed,
    "epochs": epochs,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "logging_steps": logging_steps,
    "warmup_steps": warmup_steps,
    "learning_rate": learning_rate,
    "lr_scheduler_type": lr_scheduler_type,
    "optim": optim,
    "evaluation_strategy": evaluation_strategy,
    "eval_steps": eval_steps,
    "save_strategy": save_strategy,
    "save_steps": save_steps,
    "load_best_model_at_end": load_best_model_at_end,
    "metric_for_best_model": metric_for_best_model,
    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_dir + "/training_config.json", "w") as f:
    json.dump(training_config, f, indent=4)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration for a hypothetical 1B parameter model
config_1B = LlamaConfig(
    vocab_size=32000,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=hidden_layers,
    num_attention_heads=attention_heads,
    max_position_embeddings=context_length,
    pad_token_id=2,
    torch_dtype="bfloat16"
)

# Initialize the model with bfloat16 precision
model = CustomLlamaModel(config_1B)
# model = model.half()  # Convert model parameters to bfloat16
model = model.to(device)  # Move model to GPU
model = model.train()  # Set model to training mode

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token

# Prepare dataset
print("Loading the dataset...")
dataset = load_dataset(dataset_path, dataset_config)

# Select the first dataset_size examples from the training set
if dataset_size > 0:
    print("Selecting the first", dataset_size, "examples from the dataset...")
    dataset = dataset["train"].select(range(dataset_size))
else:
    dataset_size = len(dataset["train"])
    print("Using the entire dataset of size", dataset_size)
    dataset = dataset["train"]

# Shuffling the dataset
print("Shuffling the dataset...")
dataset = dataset.shuffle(seed=seed)

# Split the dataset into training and evaluation sets (dataset_split% for training, 1-dataset_split% for evaluation)
print("Splitting the dataset into training and evaluation sets...")
print("Training set size:", int(dataset_size * dataset_split))
print("Evaluation set size:", dataset_size - int(dataset_size * dataset_split))
dataset = dataset.train_test_split(test_size=1-dataset_split, seed=seed)


# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize all texts and return overflow tokens as separate examples
    tokenized_batches = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        stride=stride,
        return_tensors="pt"
    )

    # Shift the input ids to the left to create the labels so that the model predicts the next token.
    # The label for the last token is set to -100, so it's ignored by the loss function.
    tokenized_batches["labels"] = tokenized_batches.input_ids.clone()
    tokenized_batches["labels"][:, :-1] = tokenized_batches["labels"][:, 1:].clone()
    tokenized_batches["labels"][:, -1] = -100

    return Dataset.from_dict(tokenized_batches)


# Tokenize the training and evaluation sets
print("Tokenizing the dataset...")
print("Tokenizing the training set...")
tokenized_train = tokenize_function(dataset["train"])
print("Tokenizing the evaluation set...")
tokenized_eval = tokenize_function(dataset["test"])

# TrainingArguments setup
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    logging_dir=logging_dir,
    report_to=["tensorboard"],
    logging_steps=logging_steps,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=True,  # Enable mixed-precision training
    bf16_full_eval=True,  # Enable mixed-precision evaluation
    optim=optim,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Count the number of parameters in the model and print it in billions (B) or millions (M), if applicable
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params/1e9:.2f}B" if num_params >
      1e9 else f"Number of parameters: {num_params/1e6:.2f}M")

# Start training
print("Starting training...")
trainer.train()

# Save the trained model
print("Saving the trained model...")
model.save_pretrained(output_dir)
