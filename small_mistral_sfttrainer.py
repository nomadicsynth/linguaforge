from datasets import load_dataset, DatasetDict
import json
import os
import torch
from transformers import (
    MistralForCausalLM,
    MistralConfig,
    AutoTokenizer,
    TrainingArguments,
)
from trl import set_seed, SFTTrainer

hf_token = "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"

# Model settings
hidden_layers = 12  # Number of transformer layers
hidden_size = 2048  # Size of the hidden states in the transformer layers
intermediate_size = 4096  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
attn_dropout = 0.1  # Dropout rate for the attention probabilities
context_length = 2048  # Maximum sequence length
template_model_name = "mistralai/Mistral-7B-v0.1"  # Name of the model to use as a template

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = 100000  # Number of examples to use from the dataset
dataset_split = 0.9  # Percentage of examples to use for training

# Training settings
seed = 42
learning_rate = 9.8e-5  # Learning rate for the AdamW optimizer
lr_scheduler_type = "cosine_with_restarts"  # Use a cosine annealing learning rate scheduler
num_train_epochs = 5
per_device_train_batch_size = 1
warmup_ratio = 0.15
weight_decay = 0.01  # Weight decay for the AdamW optimizer
gradient_accumulation_steps = 2  # Number of gradient accumulation steps
optim = "adamw_torch"  # Use PyTorch's AdamW optimizer
results_dir = "./results/run-3"  # Directory to save the results

# Set seed for reproducibility
set_seed(seed)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No CUDA device found. Please use a CUDA-enabled device for training.")

print(f"Using device: {device}")

# Configuration for a hypothetical <1B parameter model
config_1B = MistralConfig().from_pretrained(template_model_name, token=hf_token)
config_1B.hidden_size = hidden_size
config_1B.intermediate_size = intermediate_size
config_1B.num_hidden_layers = hidden_layers
config_1B.num_attention_heads = attention_heads
config_1B.max_position_embeddings = context_length
config_1B.sliding_window = context_length,
config_1B.pad_token_id = config_1B.eos_token_id
config_1B.torch_dtype = "bfloat16"
config_1B.attn_implementation = "flash_attention_2"

# Load tokenizer
print(f"Loading the tokenizer from {template_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(template_model_name, token=hf_token)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token
tokenizer.padding_side = 'right'

# Set stride for splitting the input into multiple sequences
tokenizer.model_max_length = context_length
tokenizer.stride = stride  # Probably doesn't work. Investigate later.

# Load the dataset
print(f"Loading the dataset from {dataset_name} ({dataset_config})...")
dataset = load_dataset(dataset_path, dataset_config)


# Define the model initialization function
def model_init(model_config: MistralConfig) -> MistralForCausalLM:
    print("Initializing the model with the following configuration:")
    for key, value in model_config.__dict__.items():
        print(f"  {key}: {value}")

    model = MistralForCausalLM(model_config).to(device)
    return model


def run_training(
        dataset: DatasetDict,
        model_config: MistralConfig,
        learning_rate: float = 9.8e-5,
        lr_scheduler_type: str = "linear",
        num_train_epochs: int = 5,
        per_device_train_batch_size: int = 2,
        warmup_ratio: float = 0.150,
        gradient_accumulation_steps: int = 1,
        dataset_size: int = 5000,
        results_dir: str = "./results",
        dataset_split: float = 0.9,
        optim: str = "adamw_torch",
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        seed: int = 42,
):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # TrainingArguments setup
    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="epoch",
        eval_steps=0.5 / num_train_epochs,
        logging_dir=f"{results_dir}/logs/",
        logging_strategy="steps",
        logging_steps=100,
        report_to="tensorboard",
        optim="adamw_torch",
        save_strategy="epoch",
        bf16=True,  # Enable mixed-precision training
        bf16_full_eval=True,  # Enable mixed-precision evaluation
        seed=seed,
    )

    # Prepare the dataset
    dataset_train, dataset_eval = prepare_dataset(dataset, dataset_size, dataset_split, seed)

    # Set the dropout rate
    model_config.attention_dropout = dropout

    # Initialize the model
    model = model_init(model_config)

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        dataset_text_field="text",
        packing=True,
        max_seq_length=model_config.max_position_embeddings,
        tokenizer=tokenizer,
    )

    # Print the model size with suffix 'G' or 'M'
    model_size = sum(p.numel() for p in trainer.model.parameters())
    model_size = model_size / 1e9 if model_size > 1e9 else model_size / 1e6
    model_size_suffix = "G" if model_size > 1e3 else "M"

    dataset_train_size = len(dataset_train)
    dataset_eval_size = len(dataset_eval)

    dataset,
    learning_rate,
    lr_scheduler_type,
    num_train_epochs,
    per_device_train_batch_size,
    warmup_ratio,
    gradient_accumulation_steps,
    dataset_size,
    results_dir,
    dataset_split,
    optim

    # Print the hyperparameters
    print("Hyperparameters:")
    print(f"  Dataset: {dataset_name} ({dataset_config})")
    print(f"  Model size: {model_size:.2f}{model_size_suffix} parameters")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Learning rate scheduler: {lr_scheduler_type}")
    print(f"  Per-device train batch size: {per_device_train_batch_size}")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Training set size: {dataset_train_size}")
    print(f"  Evaluation set size: {dataset_eval_size}")
    print(f"  Results directory: {results_dir}")
    print(f"  Optimizer: {optim}")
    print()

    # Save all the details to a JSON file in the results directory
    with open(f"{results_dir}/details.json", "w") as f:
        json.dump(
            {
                "dataset": f"{dataset_name} ({dataset_config})",
                "model_size": f"{model_size:.2f}{model_size_suffix}",
                "learning_rate": learning_rate,
                "lr_scheduler_type": lr_scheduler_type,
                "per_device_train_batch_size": per_device_train_batch_size,
                "epochs": num_train_epochs,
                "warmup_ratio": warmup_ratio,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "dataset_train_size": dataset_train_size,
                "dataset_eval_size": dataset_eval_size,
                "results_dir": results_dir,
                "optim": optim,
            },
            f,
        )

    # Train the model
    trainer.train()


def prepare_dataset(dataset: DatasetDict, dataset_size: int, dataset_split: float, seed: int = 42):
    prepared_dataset = None
    # Select the first dataset_size examples from the training set
    if dataset_size > 0:
        print("Selecting the first", dataset_size, "examples from the dataset...")
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
    return prepared_dataset["train"], prepared_dataset["test"]


# Start the training
if __name__ == "__main__":
    run_training(
        dataset=dataset,
        model_config=config_1B,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataset_size=dataset_size,
        results_dir=results_dir,
        dataset_split=dataset_split,
        optim=optim,
        weight_decay=weight_decay,
        dropout=attn_dropout,
        seed=seed,
    )
