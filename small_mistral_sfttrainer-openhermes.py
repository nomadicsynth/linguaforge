# Set the CUDA_VISIBLE_DEVICES environment variable before importing torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

from datasets import load_dataset, DatasetDict
import json
import os
import time
import torch
from torch import nn
from transformers import (
    MistralForCausalLM,
    AutoTokenizer,
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

# Ignore the warning that starts with "Token indices sequence length is longer than the specified maximum sequence length for this model"
warnings.filterwarnings(
    'ignore',
    'Token indices sequence length is longer than the '
    'specified maximum sequence length for this model.+',
    append=True
)


# The model to finetune
model_path = "/media/gronkomatic/Embiggen/ai-stuff/training-results/runs/run-20240315-211134/checkpoint-56144"
context_length = 2048  # Maximum sequence length

# Dataset settings
dataset_name = "teknium/OpenHermes-2.5"  # Name of the dataset to use
dataset_config = "default"  # Configuration of the dataset to use
dataset_path = "/media/gronkomatic/Embiggen/ai-stuff/datasets/OpenHermes-2.5-chatML"  # Path to the dataset
dataset_size = 0  # Number of examples to use from the dataset
dataset_split = 0.9  # Percentage of examples to use for training

# Training settings
results_dir = f"/media/gronkomatic/Embiggen/ai-stuff/training-results/runs/run-{time.strftime('%Y%m%d-%H%M%S')}"  # Directory to save the results
seed = 42  # Random seed for reproducibility
learning_rate = 3.66e-4  # Learning rate for the AdamW optimizer
lr_scheduler_type = "cosine"  # Use a cosine annealing learning rate scheduler
num_train_epochs = 1  # Number of training epochs
per_device_train_batch_size = 3  # Batch size per GPU/TPU core/CPU for training
warmup_ratio = 0.10  # Ratio of the number of warmup steps to the total number of training steps
weight_decay = 0.01  # Weight decay for the AdamW optimizer
max_grad_norm = 1.0  # Maximum gradient norm
gradient_accumulation_steps = 1  # Number of steps to accumulate gradients for
gradient_checkpointing = False  # Causes a segfault when enabled
# Choose the optimizer to use
# 'adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla',
# 'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision',
# 'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit',
# 'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit',
# 'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit', 'rmsprop_bnb_32bit', 'galore_adamw',
# 'galore_adamw_8bit', 'galore_adafactor', 'galore_adamw_layerwise',
# 'galore_adamw_8bit_layerwise', 'galore_adafactor_layerwise'
optim = "adamw_bnb_8bit"

# Set seed for reproducibility
set_seed(seed)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No CUDA device found. Please use a CUDA-enabled device for training.")

print(f"Using device: {device}")

# Load tokenizer
print(f"Loading the tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token
tokenizer.padding_side = 'right'

# Set stride for splitting the input into multiple sequences
tokenizer.model_max_length = context_length
# Add chatML tokens to the tokenizer
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]},
    replace_additional_special_tokens=False
)

# Set the chat template
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# Load the dataset
print(f"Loading the dataset from {dataset_path} ({dataset_config})...")
dataset = load_dataset(dataset_path, dataset_config)
dataset = dataset.shuffle()


def run_training(
        dataset: DatasetDict,
        model_path: str,
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
        evaluation_strategy="steps",
        eval_steps=0.25 / num_train_epochs,
        logging_dir=f"{results_dir}/logs/",
        logging_strategy="steps",
        logging_steps=100,
        report_to="tensorboard",
        optim=optim,
        save_strategy="steps",
        save_steps=0.25 / num_train_epochs,
        bf16=True,  # Enable mixed-precision training
        bf16_full_eval=True,  # Enable mixed-precision evaluation
        seed=seed,
    )

    # Prepare the dataset
    dataset_train, dataset_eval = prepare_dataset(dataset, dataset_size, dataset_split, seed)

    # Initialize the model
    model = model_init(model_path)

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        dataset_text_field="text",
        packing=True,
        max_seq_length=context_length,
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
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "dataset_train_size": dataset_train_size,
                "dataset_eval_size": dataset_eval_size,
                # Model settings
                "model_path": model_path,
                "context_length": context_length,
                # Dataset settings
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "dataset_path": dataset_path,
                "dataset_size": dataset_size,
                "dataset_split": dataset_split,
                # Training settings
                "seed": seed,
                "optim": optim,
            },
            f,
        )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(f"{results_dir}/model")


# Define the model initialization function
def model_init(model_path: str) -> MistralForCausalLM:
    model = MistralForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = model.to(device)

    return model


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
        model_path=model_path,
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
        seed=seed,
    )
