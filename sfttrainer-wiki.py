import os

# Set the CUDA_VISIBLE_DEVICES environment variable before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

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

# Use Mistral-7B-v0.1 as a template for the model settings
template_model_name = "mistralai/Mistral-7B-v0.1"

# Model settings
hidden_layers = 1  # Number of transformer layers
hidden_size = 2048  # Size of the hidden states in the transformer layers
intermediate_size = 4096  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
context_length = 1024  # Maximum sequence length

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = 500  # Number of examples to use from the dataset
dataset_split = 0.9  # Percentage of examples to use for training
stride = 150  # Stride for splitting the input into multiple sequences. Doesn't work with Mistral according to CoPilot, but what would they know?

# Directory to save the results
results_dir = f"/media/gronkomatic/Embiggen/ai-stuff/training-results/runs/run-{time.strftime('%Y%m%d-%H%M%S')}"

# Training settings
seed = 42  # Random seed for reproducibility
dtype = "bfloat16"  # Data type to use for the model
learning_rate = 8.6e-4  # Learning rate for the AdamW optimizer
lr_scheduler_type = "linear"  # Use a cosine annealing learning rate scheduler
num_train_epochs = 1  # Number of training epochs
per_device_train_batch_size = 14  # Batch size per GPU/TPU core/CPU for training
gradient_accumulation_steps = 1  # Number of steps to accumulate gradients for
warmup_ratio = 0.10  # Ratio of the number of warmup steps to the total number of training steps
weight_decay = 0.0434  # Weight decay for the AdamW optimizer
max_grad_norm = 1.0  # Maximum gradient norm
gradient_checkpointing = False  # Causes a segfault when enabled
# Choose the optimizer to use
# 'adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla',
# 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision',
# 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit',
# 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit',
# 'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit', 'rmsprop_bnb_32bit', 'galore_adamw',
# 'galore_adamw_8bit', 'galore_adafactor', 'galore_adamw_layerwise',
# 'galore_adamw_8bit_layerwise', 'galore_adafactor_layerwise'
optim = "adamw_8bit"

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

# Configuration for the model
template_model_config = MistralConfig.from_pretrained(template_model_name)

model_config = dict(
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=hidden_layers,
    num_attention_heads=attention_heads,
    num_key_value_heads=1,  # Enables Multi-Query Attention (MQA)
    max_position_embeddings=4096 * 32,
    use_cache=False if gradient_checkpointing else True,
    pad_token_id=template_model_config.pad_token_id,
    sliding_window=context_length,
    torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32,
    attn_implementation="flash_attention_2"
)
model_config = MistralConfig(**model_config)

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


# TrainingArguments setup
training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    max_grad_norm=max_grad_norm,
    warmup_steps=500,
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

# Train the model
trainer.train()

# Save the model
model_path = f"{results_dir}/model"
trainer.save_model(model_path)
print(f"Model saved to {model_path}")

# TODO: Evaluate the model
# results = trainer.evaluate()
