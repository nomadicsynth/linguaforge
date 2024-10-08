"""
Pause Training
Untested code to use "Pause Tokens" to give the model more thinking time.

Based on the paper "Think before you speak: Training Language Models With Pause Tokens", 
by Sachin Goyal,Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, Vaishnavh Nagarajan
https://arxiv.org/abs/2310.02226

WIP: This code is untested and may not even be complete. Maybe I'll get around to it one day.
"""

import os

# Set the CUDA_VISIBLE_DEVICES environment variable before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

import argparse
from datasets import load_dataset, DatasetDict
import json
import optuna
import time
import torch
from transformers import (
    AutoTokenizer,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedModel,
    TrainingArguments,
)
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy
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

# Add the argument for the project name
parser.add_argument("--project_name", type=str, default="mistral-wiki", help="Name of the project")

# Add the argument for the results directory
parser.add_argument("--output_dir", type=str,
                    default=f"/media/gronkomatic/Embiggen/ai-stuff/training-results", help="Directory to save the results")

# Add the arguments for the model settings
parser.add_argument("--template_model_name", type=str, default="mistralai/Mistral-7B-v0.1", help="Template model name")
parser.add_argument("--hidden_layers", type=int, default=1, help="Number of transformer layers")
parser.add_argument("--hidden_size", type=int, default=2048, help="Size of the hidden states in the transformer layers")
parser.add_argument("--intermediate_size", type=int, default=4096,
                    help="Size of the feed-forward network in the transformer layers")
parser.add_argument("--attention_heads", type=int, default=32, help="Number of attention heads")
parser.add_argument("--context_length", type=int, default=1024, help="Maximum sequence length")

# Add the arguments for the dataset settings
parser.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia", help="Name of the dataset to use")
parser.add_argument("--dataset_config", type=str, default="20231101.en", help="Configuration of the dataset to use")
parser.add_argument("--dataset_path", type=str,
                    default="/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia", help="Path to the dataset")
parser.add_argument("--dataset_size", type=int, default=500, help="Number of examples to use from the dataset")
parser.add_argument("--dataset_split", type=float, default=0.9, help="Percentage of examples to use for training")
parser.add_argument("--stride", type=int, default=150, help="Stride for splitting the input into multiple sequences")

# Add the arguments for the training settings
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--dtype", type=str, default="bfloat16",
                    help="Data type to use for the model",
                    choices=["float16", "bfloat16", "float32"])
parser.add_argument("--learning_rate", type=float, default=8.6e-4, help="Learning rate for the AdamW optimizer")
parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=14,
                    help="Batch size per GPU/TPU core/CPU for training")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of steps to accumulate gradients for")
parser.add_argument("--warmup_ratio", type=float, default=0.10,
                    help="Ratio of the number of warmup steps to the total number of training steps")
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
parser.add_argument("--weight_decay", type=float, default=0.0434, help="Weight decay for the AdamW optimizer")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer to use")

# Add the arguments for the Optuna study
parser.add_argument("--run_hyperparameter_search", action="store_true", help="Enable hyperparameter search")
parser.add_argument("--study_name", type=str, default=f"hyperparameter_search", help="Name of the Optuna study")
parser.add_argument("--n_trials", type=int, default=1, help="Number of hyperparameter search trials")
parser.add_argument("--opt_lr", action="store_true", help="Optimize the learning rate")
parser.add_argument("--lr_range", type=float, nargs=2,
                    default=[1e-5, 1e-2], help="Range of learning rates to use for hyperparameter search")
parser.add_argument("--opt_dtype", action="store_true", help="Optimize the data type")
parser.add_argument("--dtype_categorical", type=str, nargs="+",
                    default=["float16", "bfloat16"], help="Categorical values for the data type to use")
parser.add_argument("--opt_lr_scheduler_type", action="store_true", help="Optimize the learning rate scheduler type")
parser.add_argument("--lr_scheduler_types", type=str, nargs="+", default=[
                    "linear", "cosine", "cosine_with_restarts", "polynomial"], help="Categorical values for the learning rate scheduler type")
parser.add_argument("--opt_attention_heads", action="store_true", help="Optimize the number of attention heads")
parser.add_argument("--attention_heads_categorical", type=int, nargs="+",
                    default=[8, 16, 32, 64], help="Categorical values for the number of attention heads")
parser.add_argument("--opt_train_epochs", action="store_true", help="Optimize the number of training epochs")
parser.add_argument("--train_epochs_range", type=int, nargs=2,
                    default=[1, 7], help="Range of training epochs to use for hyperparameter search")
parser.add_argument("--opt_per_device_train_batch_size", action="store_true", help="Optimize the batch size per device")
parser.add_argument("--per_device_train_batch_size_range", type=int, nargs=2,
                    default=[1, 6], help="Range of batch sizes to use for hyperparameter search")
parser.add_argument("--opt_gradient_accumulation_steps", action="store_true",
                    help="Optimize the number of gradient accumulation steps")
parser.add_argument("--gradient_accumulation_steps_categorical", type=int, nargs="+",
                    default=[1, 2, 4, 8], help="Categorical values for the number of gradient accumulation steps")
parser.add_argument("--opt_weight_decay", action="store_true", help="Optimize the weight decay")
parser.add_argument("--weight_decay_range", type=float, nargs=2,
                    default=[0.0, 0.1], help="Range of weight decay values to use for hyperparameter search")
parser.add_argument("--opt_max_grad_norm", action="store_true", help="Optimize the maximum gradient norm")
parser.add_argument("--max_grad_norm_range", type=float, nargs=2,
                    default=[0.5, 1.5], help="Range of maximum gradient norms to use for hyperparameter search")
parser.add_argument("--opt_warmup_ratio", action="store_true", help="Optimize the warmup ratio")
parser.add_argument("--warmup_ratio_range", type=float, nargs=2,
                    default=[0.1, 0.2], help="Range of warmup ratios to use for hyperparameter search")
parser.add_argument("--opt_hidden_layers", action="store_true", help="Optimize the number of hidden layers")
parser.add_argument("--hidden_layers_range", type=int, nargs=2,
                    default=[1, 18], help="Range of hidden layers to use for hyperparameter search")

args = parser.parse_args()

timestamp = time.strftime("%Y%m%d-%H%M%S")
output_dir = args.output_dir + "/" + args.project_name

# Use Mistral-7B-v0.1 as a template for the model settings
template_model_name = args.template_model_name

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
results_dir = f"{output_dir}/training-run-{timestamp}"

# Training settings
seed = args.seed  # Random seed for reproducibility
dtype = args.dtype  # Data type to use for the model
# Set dtype to the appropriate torch dtype
dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32
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

# Optuna study settings
run_hyperparameter_search = args.run_hyperparameter_search  # Enable hyperparameter search
study_name = args.study_name  # Name of the Optuna study
study_dir = f"{output_dir}/optuna-study-{timestamp}"
n_trials = args.n_trials  # Number of hyperparameter search trials
lr_range = args.lr_range  # Range of learning rates to use for hyperparameter search
dtype_categorical = args.dtype_categorical  # Categorical values for the data type to use
# Categorical values for the learning rate scheduler type
lr_scheduler_types = args.lr_scheduler_types
attention_heads_categorical = args.attention_heads_categorical  # Categorical values for the number of attention heads
train_epochs_range = args.train_epochs_range  # Range of training epochs to use for hyperparameter search
warmup_ratio_range = args.warmup_ratio_range  # Range of warmup ratios to use for hyperparameter search
# Range of batch sizes to use for hyperparameter search
per_device_train_batch_size_range = args.per_device_train_batch_size_range
# Categorical values for the number of gradient accumulation steps
gradient_accumulation_steps_categorical = args.gradient_accumulation_steps_categorical
weight_decay_range = args.weight_decay_range  # Range of weight decay values to use for hyperparameter search
max_grad_norm_range = args.max_grad_norm_range  # Range of maximum gradient norms to use for hyperparameter search
hidden_layers_range = args.hidden_layers_range  # Range of hidden layers to use for hyperparameter search

# Set the final output directory
if run_hyperparameter_search:
    results_dir = study_dir

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
tokenizer.model_max_length = context_length

# Set up truncation and padding
tokenizer.set_truncation_and_padding(
    padding_strategy=PaddingStrategy.LONGEST,
    truncation_strategy=TruncationStrategy.LONGEST_FIRST,
    max_length=context_length,
    stride=stride,
    pad_to_multiple_of=8
)

# Load the dataset
print(f"Loading the dataset from {dataset_name} ({dataset_config})...")
dataset = load_dataset(dataset_path, dataset_config)

model_config = MistralConfig(
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=hidden_layers,
    num_attention_heads=attention_heads,
    num_key_value_heads=1,  # Enables Multi-Query Attention (MQA)
    max_position_embeddings=context_length,
    use_cache=False if gradient_checkpointing else True,
    pad_token_id=tokenizer.pad_token_id,
    sliding_window=None,
    torch_dtype=dtype,
    attn_implementation="flash_attention_2"
)


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


# Hyperparameter search objective function
def compute_objective(metrics: dict) -> float:
    return metrics["eval_loss"]


# Hyperparameter search space
def hp_space(trial: optuna.Trial) -> dict:
    space = {}
    if args.opt_lr:
        space["learning_rate"] = trial.suggest_float("learning_rate", lr_range[0], lr_range[1])
    if args.opt_dtype:
        space["dtype"] = trial.suggest_categorical("dtype", dtype_categorical)
    if args.opt_lr_scheduler_type:
        space["lr_scheduler_type"] = trial.suggest_categorical("lr_scheduler_type", lr_scheduler_types)
    if args.opt_attention_heads:
        space["attention_heads"] = trial.suggest_categorical("attention_heads", attention_heads_categorical)
    if args.opt_train_epochs:
        space["num_train_epochs"] = trial.suggest_int("num_train_epochs", train_epochs_range[0], train_epochs_range[1])
    if args.opt_per_device_train_batch_size:
        space["per_device_train_batch_size"] = trial.suggest_int(
            "per_device_train_batch_size", per_device_train_batch_size_range[0], per_device_train_batch_size_range[1])
    if args.opt_gradient_accumulation_steps:
        space["gradient_accumulation_steps"] = trial.suggest_categorical(
            "gradient_accumulation_steps", gradient_accumulation_steps_categorical)
    if args.opt_weight_decay:
        space["weight_decay"] = trial.suggest_float("weight_decay", weight_decay_range[0], weight_decay_range[1])
    if args.opt_max_grad_norm:
        space["max_grad_norm"] = trial.suggest_float("max_grad_norm", max_grad_norm_range[0], max_grad_norm_range[1])
    if args.opt_warmup_ratio:
        space["warmup_ratio"] = trial.suggest_float("warmup_ratio", warmup_ratio_range[0], warmup_ratio_range[1])
    if args.opt_hidden_layers:
        space["hidden_layers"] = trial.suggest_int("hidden_layers", hidden_layers_range[0], hidden_layers_range[1])

    return space


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


def save_model(path: str) -> str:
    model_path = f"{path}/model"
    trainer.save_model(model_path)
    # print(f"Model saved to {model_path}")

    return model_path

# DeepSpeed settings - uncomment to enable
# deepspeed_config = {
#     "zero_optimization": {
#         "stage": 1
#     }
# }

# TrainingArguments setup
training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    optim=optim,
    weight_decay=weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{results_dir}/logs/",
    logging_strategy="steps",
    logging_steps=0.1 / num_train_epochs,
    load_best_model_at_end=True,
    seed=seed,
    bf16=(dtype == torch.bfloat16),
    bf16_full_eval=(dtype == torch.bfloat16),
    fp16=(dtype == torch.float16),
    fp16_full_eval=(dtype == torch.float16),
    report_to="none" if run_hyperparameter_search else "tensorboard",
    # deepspeed=deepspeed_config,
)

# Prepare the dataset
prepared_dataset = prepare_dataset(dataset, dataset_size, dataset_split)

# Save the prepared dataset
prepared_dataset.save_to_disk(f"{results_dir}/dataset/")
print("Prepared dataset saved to", f"{results_dir}/dataset/")

# Save the dataset configuration
with open(f"{results_dir}/dataset_config.json", "w") as f:
    json.dump({"dataset_name": dataset_name, "dataset_config": dataset_config,
              "dataset_size": dataset_size, "dataset_split": dataset_split, "stride": stride}, f, indent=2)

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


def run_training():
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
        save_model(results_dir)
        print("Training progress saved.")
        print("Training interrupted by user.")
        exit()

    print("Training complete!")
    print()

    # Save the model
    model_path = save_model(results_dir)

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


def run_study():
    study_db_path = f"{results_dir}/optuna.db"
    study_storage = f"sqlite:///{study_db_path}"

    optuna_kwargs = {
        "study_name": study_name,
        "storage": study_storage,
    }

    # Run the hyperparameter search
    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,
        compute_objective=compute_objective,
        n_trials=n_trials,
        direction="minimize",
        backend="optuna",
        **optuna_kwargs,
    )

    # Load the study from the database
    study = optuna.load_study(study_name=study_name, storage=study_storage)

    # Visualize the study, saving the plots to the study directory
    vis_dir = f"{results_dir}/study-visualizations/"
    os.makedirs(vis_dir, exist_ok=True)
    optuna.visualization.plot_optimization_history(study).write_html(f"{vis_dir}/optimization_history.html")
    if len(study.trials) > 1:
        optuna.visualization.plot_parallel_coordinate(study).write_html(f"{vis_dir}/parallel_coordinate.html")
    optuna.visualization.plot_slice(study).write_html(f"{vis_dir}/slice.html")
    optuna.visualization.plot_contour(study).write_html(f"{vis_dir}/contour.html")
    optuna.visualization.plot_parallel_coordinate(study).write_html(f"{vis_dir}/parallel_coordinate.html")
    optuna.visualization.plot_edf(study).write_html(f"{vis_dir}/edf.html")

    # Save the best run
    best_run_path = f"{results_dir}/best_run.json"
    with open(best_run_path, "w") as f:
        json.dump(best_run, f, indent=4)
    print(f"Best run saved to {best_run_path}")

    # Save the best model
    best_model_path = save_model(results_dir)

    # Print the best run
    print("Best run:")
    print(json.dumps(best_run, indent=4))

    # Display the results
    print("Results directory:", results_dir)
    print("Best run saved to:", best_run_path)
    print("Study saved to:", study_db_path)
    print("Visualizations saved to:", vis_dir)
    print("Best model saved to:", best_model_path)
    print("Logs saved to:", f"{results_dir}/logs/")
    print()
    print("To view the training logs, run the following command:")
    print(f"tensorboard --logdir {results_dir}/logs/")


# Run whatever is needed
if run_hyperparameter_search:
    run_study()
else:
    run_training()
