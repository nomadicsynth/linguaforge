import os

# Set the CUDA_VISIBLE_DEVICES environment variable before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

# Set the DS_SKIP_CUDA_CHECK environment variable before importing deepspeed
# os.environ["DS_SKIP_CUDA_CHECK"] = "1"

from datasets import load_dataset, DatasetDict
import json
import optuna
import pickle
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

# Ignore the warning that starts with "Token indices sequence length is longer than the specified maximum sequence length for this model"
warnings.filterwarnings(
    'ignore',
    'Token indices sequence length is longer than the '
    'specified maximum sequence length for this model.+',
    append=True
)

# Use Mistral-7B-v0.1 as a template for the model settings
template_model_name = "mistralai/Mistral-7B-v0.1"

# Model settings
hidden_layers = 1  # Number of transformer layers
hidden_size = 2048  # Size of the hidden states in the transformer layers
intermediate_size = 4096  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
attn_dropout = 0.0  # Dropout rate for the attention probabilities
context_length = 1024  # Maximum sequence length

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = 500  # Number of examples to use from the dataset
dataset_split = 0.9  # Percentage of examples to use for training
stride = 50  # Stride for splitting the input into multiple sequences. Doesn't work with Mistral according to CoPilot, but what would they know?

# Training settings
seed = 42  # Random seed for reproducibility
dtype = "bfloat16"  # Data type to use for the model
learning_rate = 8.2e-4  # Learning rate for the AdamW optimizer
lr_scheduler_type = "linear"  # Use a cosine annealing learning rate scheduler
num_train_epochs = 2  # Number of training epochs
per_device_train_batch_size = 14  # Batch size per GPU/TPU core/CPU for training
gradient_accumulation_steps = 1  # Number of steps to accumulate gradients for
warmup_ratio = 0.10  # Ratio of the number of warmup steps to the total number of training steps
weight_decay = 0.06388269955610547  # Weight decay for the AdamW optimizer
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

# Optuna study settings
study_timestamp = time.strftime("%Y%m%d-%H%M%S")
study_name = f"mistral-small_hyperparameter_search-{study_timestamp}"
study_dir = f"/media/gronkomatic/Embiggen/ai-stuff/training-results/studies/{study_name}"
n_trials = 10  # Number of hyperparameter search trials
lr_range = [1e-6, 1.4e-3]  # Range of learning rates to use for hyperparameter search
dtype_categorical = ["float16", "bfloat16"]  # Categorical values for the data type to use
dataset_size_categorical = [1000, 2000, 3000]  # Categorical values for the number of examples to use from the dataset
# Categorical values for the learning rate scheduler type
lr_scheduler_types = ["linear", "cosine", "cosine_with_restarts", "polynomial"]
attention_heads_categorical = [8, 16, 32, 64]  # Categorical values for the number of attention heads
train_epochs_range = [1, 7]  # Range of training epochs to use for hyperparameter search
warmup_ratio_range = [0.1, 0.2]  # Range of warmup ratios to use for hyperparameter search
# Categorical values for the number of gradient accumulation steps
per_device_train_batch_size_range = [1, 6]  # Range of batch sizes to use for hyperparameter search
gradient_accumulation_steps_categorical = [1, 2, 4, 8]
attn_dropout_range = [0.0, 0.2]  # Range of attention dropout rates to use for hyperparameter search
weight_decay_range = [0.0, 0.1]  # Range of weight decay values to use for hyperparameter search
max_grad_norm_range = [0.5, 1.5]  # Range of maximum gradient norms to use for hyperparameter search
hidden_layers_range = [1, 18]  # Range of hidden layers to use for hyperparameter search

# Set seed for reproducibility
set_seed(seed)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No CUDA device found. Please use a CUDA-enabled device for training.")

print(f"Using device: {device}")

# Configuration for the model
template_model_config = MistralConfig.from_pretrained(template_model_name)

# with deepspeed.zero.Init():
model_config = dict(
    hidden_size = hidden_size,
    intermediate_size = intermediate_size,
    num_hidden_layers = hidden_layers,
    num_attention_heads = attention_heads,
    num_key_value_heads = 1,  # Enables Multi-Query Attention (MQA)
    max_position_embeddings = 4096 * 32,
    use_cache = False if gradient_checkpointing else True,
    pad_token_id = template_model_config.pad_token_id,
    sliding_window = context_length,
    attention_dropout = attn_dropout,
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32,
    attn_implementation = "flash_attention_2"
)

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
def prepare_dataset(dataset: DatasetDict, dataset_size: int, dataset_split: float) -> DatasetDict:
    print("Preparing the dataset...")
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
    return prepared_dataset


# Hyperparameter search objective function
def compute_objective(metrics: dict) -> float:
    return metrics["eval_loss"]


# Hyperparameter search space
def hp_space(trial: optuna.Trial) -> dict:
    return {
        # "dtype": trial.suggest_categorical("dtype", dtype_categorical),
        # "attention_heads": trial.suggest_categorical("attention_heads", attention_heads_categorical),
        # "hidden_layers": trial.suggest_int("hidden_layers", hidden_layers_range[0], hidden_layers_range[1]),
        "learning_rate": trial.suggest_float("learning_rate", lr_range[0], lr_range[1]),
        # "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", lr_scheduler_types),
        # "num_train_epochs": trial.suggest_int("num_train_epochs", train_epochs_range[0], train_epochs_range[1]),
        # "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", per_device_train_batch_size_range[0], per_device_train_batch_size_range[1]),
        # "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", gradient_accumulation_steps_categorical),
        # "attn_dropout": trial.suggest_float("attn_dropout", attn_dropout_range[0], attn_dropout_range[1]),
        # "weight_decay": trial.suggest_float("weight_decay", weight_decay_range[0], weight_decay_range[1]),
        # "max_grad_norm": trial.suggest_float("max_grad_norm", max_grad_norm_range[0], max_grad_norm_range[1]),
        # "warmup_ratio": trial.suggest_float("warmup_ratio", warmup_ratio_range[0], warmup_ratio_range[1]),
        # "dataset_size": trial.suggest_categorical("dataset_size", dataset_size_categorical),
    }


# Initialize the model
def model_init() -> PreTrainedModel:
    print("Initialising the model...")
    # with deepspeed.zero.Init():
    model_config = MistralConfig(**model_config)
    model = MistralForCausalLM(model_config)

    # Set the gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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
    output_dir=f"{study_dir}",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    optim=optim,
    weight_decay=weight_decay,
    evaluation_strategy="steps",
    eval_steps=0.5 / num_train_epochs - 0.001,
    save_strategy="no",
    # save_steps=0.5 / num_train_epochs - 0.001,
    logging_dir=f"{study_dir}/logs/",
    logging_strategy="steps",
    logging_steps=min(0.1 / num_train_epochs, 100),
    report_to="none",
    # load_best_model_at_end=True,
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

optuna_kwargs = {
    "study_name": study_name,
    "storage": f"sqlite:///{study_dir}/optuna.db"
}

if not os.path.exists(study_dir):
    os.makedirs(study_dir)

# Run the hyperparameter search
best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    compute_objective=compute_objective,
    n_trials=n_trials,
    direction="minimize",
    optuna_kwargs=optuna_kwargs,
)

# Save the best run
best_run_path = f"{study_dir}/best_run.json"
with open(best_run_path, "w") as f:
    json.dump(best_run, f, indent=4)
print(f"Best run saved to {best_run_path}")

# Save the best model
best_model_path = f"{study_dir}/best_model"
trainer.save_model(best_model_path)
print(f"Best model saved to {best_model_path}")

# Save the study
study_path = f"{study_dir}/optuna_study.pkl"
with open(study_path, "wb") as f:
    pickle.dump(trainer.study, f)
print(f"Study saved to {study_path}")

# Print the best run
print("Best run:")
print(json.dumps(best_run, indent=4))

# Analyze the study
print("Study statistics: ")
print("  Number of finished trials: ", len(trainer.study.trials))
print("  Best trial:")
trial = trainer.study.best_trial
print("    Value: ", trial.value)
print("    Params: ")
for key, value in trial.params.items():
    print(f"      {key}: {value}")

print(trainer.study.trials_dataframe())

# Visualize the study, saving the plots to the study directory
optuna.visualization.plot_optimization_history(trainer.study).write_html(f"{study_dir}/plot_optimization_history.html")
optuna.visualization.plot_slice(trainer.study).write_html(f"{study_dir}/plot_slice.html")
optuna.visualization.plot_parallel_coordinate(trainer.study).write_html(f"{study_dir}/plot_parallel_coordinate.html")
optuna.visualization.plot_param_importances(trainer.study).write_html(f"{study_dir}/plot_param_importances.html")
optuna.visualization.plot_contour(trainer.study).write_html(f"{study_dir}/plot_contour.html")
optuna.visualization.plot_edf(trainer.study).write_html(f"{study_dir}/plot_edf.html")
optuna.visualization.plot_intermediate_values(trainer.study).write_html(f"{study_dir}/plot_intermediate_values.html")
print("Study visualizations saved to", study_dir)

# Save the training arguments
training_args_path = f"{study_dir}/training_args.json"
with open(training_args_path, "w") as f:
    json.dump(training_args.__dict__, f, indent=4)
print(f"Training arguments saved to {training_args_path}")

# Save the model configuration
model_config_path = f"{study_dir}/model_config.json"
with open(model_config_path, "w") as f:
    json.dump(model_config, f, indent=4)
print(f"Model configuration saved to {model_config_path}")

# Save the dataset configuration
dataset_config_path = f"{study_dir}/dataset_config.json"
with open(dataset_config_path, "w") as f:
    json.dump(dataset, f, indent=4)
print(f"Dataset configuration saved to {dataset_config_path}")

# Save the dataset
dataset_path = f"{study_dir}/dataset"
prepared_dataset.save_to_disk(dataset_path)
print(f"Dataset saved to {dataset_path}")
