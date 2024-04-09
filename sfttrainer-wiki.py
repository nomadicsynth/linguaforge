import os

# Set the CUDA_VISIBLE_DEVICES environment variable before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

import argparse
from datasets import load_dataset, DatasetDict
import json
import time
import torch
from transformers import (
    AutoTokenizer,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedModel,
    TrainingArguments,
)
from transformers.trainer_pt_utils import get_parameter_names
from trl import set_seed, SFTTrainer
import warnings

# Ignore the warning about gathering scalars
warnings.filterwarnings(
    'ignore',
    'Was asked to gather along dimension 0, but all '
    'input tensors were scalars; will instead unsqueeze '
    'and return a vector.',
    append=True
)

# Ignore the FutureWarning about passing arguments to Accelerator
warnings.filterwarnings(
    'ignore',
    "Passing the following arguments to `Accelerator` "
    "is deprecated and will be removed in version 1.0 of Accelerate:",
    category=FutureWarning,
    append=True
)

# Set up command-line arguments with argparse
parser = argparse.ArgumentParser(description="Train a Mistral model on a Wikipedia dataset.")
parser.add_argument("--hidden_layers", type=int, default=1, help="Number of transformer layers")
parser.add_argument("--hidden_size", type=int, default=2048, help="Size of the hidden states in the transformer layers")
parser.add_argument("--intermediate_size", type=int, default=4096, help="Size of the feed-forward network in the transformer layers")
parser.add_argument("--attention_heads", type=int, default=32, help="Number of attention heads")
parser.add_argument("--context_length", type=int, default=1024, help="Maximum sequence length")
parser.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia", help="Name of the dataset to use")
parser.add_argument("--dataset_config", type=str, default="20231101.en", help="Configuration of the dataset to use")
parser.add_argument("--dataset_path", type=str, default="/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia", help="Path to the dataset")
parser.add_argument("--dataset_size", type=int, default=500, help="Number of examples to use from the dataset")
parser.add_argument("--dataset_split", type=float, default=0.9, help="Percentage of examples to use for training")
parser.add_argument("--stride", type=int, default=150, help="Stride for splitting the input into multiple sequences")
parser.add_argument("--results_dir", type=str, default=f"/media/gronkomatic/Embiggen/ai-stuff/training-results/runs/run-{time.strftime('%Y%m%d-%H%M%S')}", help="Directory to save the results")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type to use for the model")
parser.add_argument("--learning_rate", type=float, default=8.6e-4, help="Learning rate for the AdamW optimizer")
parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=14, help="Batch size per GPU/TPU core/CPU for training")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients for")
parser.add_argument("--warmup_ratio", type=float, default=0.10, help="Ratio of the number of warmup steps to the total number of training steps")
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
parser.add_argument("--weight_decay", type=float, default=0.0434, help="Weight decay for the AdamW optimizer")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer to use")
args = parser.parse_args()

# Use Mistral-7B-v0.1 as a template for the model settings
template_model_name = "mistralai/Mistral-7B-v0.1"

# Model settings
hidden_layers = args.hidden_layers  # Number of transformer layers
hidden_size = args.hidden_size  # Size of the hidden states in the transformer layers
intermediate_size = args.intermediate_size  # Size of the feed-forward network in the transformer layers
attention_heads = args.attention_heads  # Number of attention heads
context_length = args.context_length  # Maximum sequence length

# Dataset settings
dataset_name = args.dataset_name  # Name of the dataset to use
dataset_config = args.dataset_config  # Configuration of the dataset to use
dataset_path = args.dataset_path  # Path to the dataset
dataset_size = args.dataset_size  # Number of examples to use from the dataset
dataset_split = args.dataset_split  # Percentage of examples to use for training
stride = args.stride  # Stride for splitting the input into multiple sequences

# Directory to save the results
results_dir = args.results_dir

# Training settings
seed = args.seed  # Random seed for reproducibility
dtype = args.dtype  # Data type to use for the model
learning_rate = args.learning_rate  # Learning rate for the AdamW optimizer
lr_scheduler_type = args.lr_scheduler_type  # Learning rate scheduler type
num_train_epochs = args.num_train_epochs  # Number of training epochs
per_device_train_batch_size = args.per_device_train_batch_size  # Batch size per GPU/TPU core/CPU for training
gradient_accumulation_steps = args.gradient_accumulation_steps  # Number of steps to accumulate gradients for
warmup_ratio = args.warmup_ratio  # Ratio of the number of warmup steps to the total number of training steps
warmup_steps = args.warmup_steps  # Number of warmup steps
weight_decay = args.weight_decay  # Weight decay for the AdamW optimizer
max_grad_norm = args.max_grad_norm  # Maximum gradient norm
gradient_checkpointing = args.gradient_checkpointing  # Enable gradient checkpointing
# Choose the optimizer to use
# 'adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla',
# 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision',
# 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit',
# 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit',
# 'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit', 'rmsprop_bnb_32bit', 'galore_adamw',
# 'galore_adamw_8bit', 'galore_adafactor', 'galore_adamw_layerwise',
# 'galore_adamw_8bit_layerwise', 'galore_adafactor_layerwise'
optim = args.optim

# Set seed for reproducibility
set_seed(seed)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No CUDA device found. Please use a CUDA-enabled device for training.")

# Create the results directory
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print(f"Using device: {device}")

# Load tokenizer
print(f"Loading the tokenizer from {template_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(template_model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token
tokenizer.padding_side = 'right'

# Set stride for splitting the input into multiple sequences
tokenizer.model_max_length = context_length
tokenizer.stride = stride

# Load the dataset
print(f"Loading the dataset from {dataset_name} ({dataset_config})...")
dataset = load_dataset(dataset_path, dataset_config)

model_config = dict(
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=hidden_layers,
    num_attention_heads=attention_heads,
    num_key_value_heads=1,  # Enables Multi-Query Attention (MQA)
    max_position_embeddings=4096 * 32,
    use_cache=False if gradient_checkpointing else True,
    pad_token_id=tokenizer.pad_token_id,
    sliding_window=context_length,
    torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32,
    attn_implementation="flash_attention_2"
)
model_config = MistralConfig(**model_config)


# Prepare the dataset
def prepare_dataset(dataset: DatasetDict, dataset_size: int, dataset_split: float, shuffle: bool = False) -> DatasetDict:
    print("Preparing the dataset...")
    prepared_dataset = None
    if shuffle:
        dataset["train"] = dataset["train"].shuffle()
    # Select the first dataset_size examples from the training set
    if dataset_size > 0:
        print("Selecting", dataset_size, "examples from the dataset...")
        prepared_dataset = dataset["train"].select(range(dataset_size))
    else:
        dataset_size = len(dataset["train"])
        print("Using the entire dataset of size", dataset_size)
        prepared_dataset = dataset["train"]

    # Split the dataset into training and evaluation sets (dataset_split% for training, 1-dataset_split% for evaluation)
    print("Splitting the dataset into training and evaluation sets...")
    print("Training set size:", round(dataset_size * dataset_split))
    print("Evaluation set size:", dataset_size - round(dataset_size * dataset_split))
    prepared_dataset = prepared_dataset.train_test_split(test_size=1-dataset_split, seed=seed)

    # Return the training and evaluation datasets
    return prepared_dataset


# Initialize the model
def model_init() -> PreTrainedModel:
    print("Initialising the model...")
    model = MistralForCausalLM(model_config)

    # If the dtype is float16 or bfloat16, convert the model to that dtype
    if model_config.torch_dtype == "float16" or model_config.torch_dtype == torch.float16:
        model = model.half()
    elif model_config.torch_dtype == "bfloat16" or model_config.torch_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)

    # Move the model to the device
    model = model.to(device)

    # Print the model size with suffix 'G' or 'M'
    model_size = sum(p.numel() for p in model.parameters())
    model_size = model_size / 1e9 if model_size > 1e9 else model_size / 1e6
    model_size = round(model_size)
    model_size_suffix = "G" if model_size > 1e3 else "M"

    print(f"Model size: {model_size}{model_size_suffix} parameters")

    return model


def save_model() -> str:
    model_path = f"{results_dir}/model"
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")

    return model_path


# TrainingArguments setup
training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    max_grad_norm=max_grad_norm,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    optim=optim,
    weight_decay=weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{results_dir}/logs/",
    logging_strategy="steps",
    logging_steps=min(0.1 / num_train_epochs, 100),
    report_to="tensorboard",
    load_best_model_at_end=True,
    seed=seed,
    bf16=(dtype == "bfloat16"),
    bf16_full_eval=(dtype == "bfloat16"),
    fp16=(dtype == "float16"),
    fp16_full_eval=(dtype == "float16"),
)

# Prepare the dataset
prepared_dataset = prepare_dataset(dataset, dataset_size, dataset_split)

# Save the prepared dataset
prepared_dataset.save_to_disk(f"{results_dir}/dataset/")
print("Prepared dataset saved to", f"{results_dir}/dataset/")

# Save the dataset configuration
with open(f"{results_dir}/dataset_config.json", "w") as f:
    json.dump({"dataset_name": dataset_name, "dataset_config": dataset_config, "dataset_size": dataset_size, "dataset_split": dataset_split, "stride": stride}, f, indent=2)

# Free up some memory
del dataset

# Initialize the trainer
trainer = SFTTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=prepared_dataset["train"],
    eval_dataset=prepared_dataset["test"],
    dataset_text_field="text",
    packing=True,
    max_seq_length=context_length,
    tokenizer=tokenizer,
)

# Print the hyperparameters
print("Hyperparameters:")
print(f"  Learning rate: {learning_rate}")
print(f"  Learning rate scheduler: {lr_scheduler_type}")
print(f"  Per-device train batch size: {per_device_train_batch_size}")
print(f"  Epochs: {num_train_epochs}")
print(f"  Warmup ratio: {warmup_ratio}")
print(f"  Attention heads: {attention_heads}")
print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"  Weight decay: {weight_decay}")
print(f"  Results directory: {results_dir}")
print(f"  Optimizer: {optim}")
print()

# Save the hyperparameters to a file
hyperparameters = {
    "learning_rate": learning_rate,
    "lr_scheduler_type": lr_scheduler_type,
    "per_device_train_batch_size": per_device_train_batch_size,
    "num_train_epochs": num_train_epochs,
    "warmup_ratio": warmup_ratio,
    "attention_heads": attention_heads,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "weight_decay": weight_decay,
    "results_dir": results_dir,
    "optim": optim,
}
with open(f"{results_dir}/hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f, indent=2)


# Train the model
try:
    trainer.train()
except KeyboardInterrupt:
    # Save the training progress
    print("\nSaving the training progress...")
    save_model()
    print("Training progress saved.")
    print("Training interrupted by user.")
    exit()

print("Training complete!")
print()

# Save the model
model_path = save_model()

# Display the results
print("Results directory:", results_dir)
print("Model saved to:", model_path)
print("Hyperparameters saved to:", f"{results_dir}/hyperparameters.json")
print("Logs saved to:", f"{results_dir}/logs/")
print()
print("To view the training logs, run the following command:")
print(f"tensorboard --logdir {results_dir}/logs/")
print()
print("You can now fine-tune the model further or use it for generating text.")

# Congratulations! Your model has been trained successfully.

# To fine-tune the model further, you can load the model using the following code:
# model = MistralForCausalLM.from_pretrained(model_path)

# To generate text using the model, you can use the following code:
# from transformers import pipeline
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# text = generator("Hello, world!", max_length=100)[0]["generated_text"]
# print(text)

# To use the model for downstream tasks, you can use the following code:
# from transformers import Trainer, TrainingArguments
# training_args = TrainingArguments(output_dir="./results")
# trainer = Trainer(model=model, args=training_args)
# trainer.train()

# To evaluate the model on a dataset, you can use the following code:
# from datasets import load_metric
# metric = load_metric("accuracy")
# predictions = model.predict(test_dataset)
# metric.compute(predictions=predictions, references=test_dataset["label"])

# To evaluate the model on BLEU score, you can use the following code:
# from datasets import load_metric
# metric = load_metric("bleu")
# predictions = model.predict(test_dataset)
# metric.compute(predictions=predictions, references=test_dataset["translation"])
