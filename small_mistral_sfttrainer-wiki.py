from datasets import load_dataset, DatasetDict
import bitsandbytes as bnb
import json
import os
import time
import torch
from torch import nn
from transformers import (
    MistralForCausalLM,
    MistralConfig,
    AutoTokenizer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import get_parameter_names
from trl import set_seed, SFTTrainer
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Use Mistral-7B-v0.1 as a template for the model settings
template_model_name = "mistralai/Mistral-7B-v0.1"

# Model settings - Model size: approx 420M parameters
hidden_layers = 8  # Number of transformer layers
hidden_size = 2048  # Size of the hidden states in the transformer layers
intermediate_size = 4096  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
attn_dropout = 0.19  # Dropout rate for the attention probabilities
context_length = 2048  # Maximum sequence length

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = int(2.5e6)  # Number of examples to use from the dataset. 0 to use the entire dataset
dataset_split = 0.99  # Percentage of examples to use for training
stride = 50  # Stride for splitting the input into multiple sequences. Doesn't work with Mistral according to CoPilot, but what would they know?

# Training settings
# Directory to save the results
results_dir = f"/media/gronkomatic/Embiggen/ai-stuff/training-results/runs/run-{time.strftime('%Y%m%d-%H%M%S')}"
seed = 42  # Random seed for reproducibility
learning_rate = 3.1e-4  # Learning rate for the AdamW optimizer
lr_scheduler_type = "polynomial"  # Type of learning rate scheduler to use
num_train_epochs = 1  # Number of training epochs
per_device_train_batch_size = 2  # Batch size per GPU/TPU core/CPU for training
warmup_ratio = 0.010  # Ratio of the number of warmup steps to the total number of training steps
weight_decay = 0.01  # Weight decay for the AdamW optimizer
max_grad_norm = 1.0  # Maximum gradient norm
gradient_accumulation_steps = 1  # Number of steps to accumulate gradients for
gradient_checkpointing = False  # Causes a segfault when enabled
optim = "adamw_torch"  # Use PyTorch's AdamW optimizer

# Set seed for reproducibility
set_seed(seed)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No CUDA device found. Please use a CUDA-enabled device for training.")

print(f"Using device: {device}")

# Configuration for the model
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
config_1B.attn_dropout = attn_dropout
config_1B.add_cross_attention = True
config_1B.cross_attention_hidden_size = 256

# Load tokenizer
print(f"Loading the tokenizer from {template_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(template_model_name, token=hf_token)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token
tokenizer.padding_side = 'right'

# Set stride for splitting the input into multiple sequences
tokenizer.model_max_length = context_length
# tokenizer.stride = stride  # Probably doesn't work. Investigate later.

# Load the dataset
print(f"Loading the dataset from {dataset_name} ({dataset_config})...")
dataset = load_dataset(dataset_path, dataset_config)


class CustomSFTTrainer(SFTTrainer):
    def create_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "params": optimizer_grouped_parameters,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "lr": self.args.learning_rate
        }

        self.optimizer = bnb.optim.Adam8bit(**optimizer_kwargs)

        return self.optimizer


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
        warmup_steps=500,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps",
        eval_steps=1/14 / num_train_epochs,
        logging_dir=f"{results_dir}/logs/",
        logging_strategy="steps",
        logging_steps=500,
        report_to="tensorboard",
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=1/14 / num_train_epochs,
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
    trainer = CustomSFTTrainer(
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

    # Print the hyperparameters
    print("Hyperparameters:")
    print(f"  Dataset: {dataset_name} ({dataset_config})")
    print(f"  Model size: {model_size:.2f}{model_size_suffix} parameters")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Learning rate scheduler: {lr_scheduler_type}")
    print(f"  Per-device train batch size: {per_device_train_batch_size}")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Attention dropout: {attn_dropout}")
    print(f"  Attention heads: {attention_heads}")
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
                "model_size": f"{model_size:.2f}{model_size_suffix}",
                "learning_rate": learning_rate,
                "lr_scheduler_type": lr_scheduler_type,
                "per_device_train_batch_size": per_device_train_batch_size,
                "epochs": num_train_epochs,
                "warmup_ratio": warmup_ratio,
                "attention_dropout": attn_dropout,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "dataset_train_size": dataset_train_size,
                "dataset_eval_size": dataset_eval_size,
                # Model settings
                "hidden_layers": hidden_layers,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "attention_heads": attention_heads,
                "attn_dropout": attn_dropout,
                "context_length": context_length,
                "template_model_name": template_model_name,
                # Dataset settings
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "dataset_path": dataset_path,
                "dataset_split": dataset_split,
                "stride": stride,
                # Training settings
                "seed": seed,
                "optim": optim,
            },
            f,
        )

    # Train the model
    trainer.train()


def prepare_dataset(dataset: DatasetDict, dataset_size: int, dataset_split: float, seed: int = 42, shuffle: bool = True):
    prepared_dataset = None
    if shuffle:
        dataset["train"] = dataset["train"].shuffle(seed=seed)
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
