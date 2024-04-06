# Set the CUDA_VISIBLE_DEVICES environment variable before importing torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

import bitsandbytes as bnb
from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv
import json
import math
import numpy as np
import optuna
import pickle
import time
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedModel,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_pt_utils import get_parameter_names
from trl import set_seed, SFTTrainer
from typing import Union
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
    'specified maximum sequence length for this model',
    append=True
)

# Load the environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Use Mistral-7B-v0.1 as a template for the model settings
template_model_name = "mistralai/Mistral-7B-v0.1"

# Model settings - Model size: approx 420M parameters
hidden_layers = 2  # Number of transformer layers
hidden_size = 1024  # Size of the hidden states in the transformer layers
intermediate_size = 2048  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
attn_dropout = 0.18486156277928165  # Dropout rate for the attention probabilities
context_length = 512  # Maximum sequence length

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = 10000  # Number of examples to use from the dataset
dataset_split = 0.9  # Percentage of examples to use for training
stride = 50  # Stride for splitting the input into multiple sequences. Doesn't work with Mistral according to CoPilot, but what would they know?

# Training settings
seed = 42  # Random seed for reproducibility
learning_rate = 1.2e-3  # Learning rate for the AdamW optimizer
lr_scheduler_type = "polynomial"  # Use a cosine annealing learning rate scheduler
num_train_epochs = 5  # Number of training epochs
per_device_train_batch_size = 12  # Batch size per GPU/TPU core/CPU for training
warmup_ratio = 0.10  # Ratio of the number of warmup steps to the total number of training steps
weight_decay = 0.06388269955610547  # Weight decay for the AdamW optimizer
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

# Optuna study settings
study_timestamp = time.strftime("%Y%m%d-%H%M%S")
study_name = f"mistral-small_hyperparameter_search-{study_timestamp}"
study_dir = f"/media/gronkomatic/Embiggen/ai-stuff/training-results/studies/{study_name}"
n_trials = 64  # Number of hyperparameter search trials
dataset_size_categorical = [1000, 1500, 2000]  # Categorical values for the number of examples to use from the dataset
lr_range = [1e-3, 2e-3]  # Range of learning rates to use for hyperparameter search
# Categorical values for the learning rate scheduler type
lr_scheduler_types = ["linear", "cosine", "cosine_with_restarts", "polynomial"]
attention_heads_categorical = [8, 16, 32, 64]  # Categorical values for the number of attention heads
train_epochs_range = [1, 7]  # Range of training epochs to use for hyperparameter search
per_device_train_batch_size_range = [1, 3]  # Range of batch sizes to use for hyperparameter search
warmup_ratio_range = [0.1, 0.2]  # Range of warmup ratios to use for hyperparameter search
# Categorical values for the number of gradient accumulation steps
gradient_accumulation_steps_categorical = [1, 2, 4, 8, 16, 32, 64]
attn_dropout_range = [0.0, 0.2]  # Range of attention dropout rates to use for hyperparameter search
weight_decay_range = [0.0, 0.1]  # Range of weight decay values to use for hyperparameter search
max_grad_norm_range = [0.5, 1.5]  # Range of maximum gradient norms to use for hyperparameter search


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

# Load tokenizer
print(f"Loading the tokenizer from {template_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(template_model_name, token=hf_token)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token
tokenizer.padding_side = 'right'

# Set stride for splitting the input into multiple sequences
tokenizer.model_max_length = context_length
tokenizer.stride = stride

# Load the dataset
print(f"Loading the dataset from {dataset_name} ({dataset_config})...")
dataset = load_dataset(dataset_path, dataset_config)


# Objective function for Optuna
class Objective(TrainerCallback):
    def __init__(self, dataset: Union[dict, DatasetDict], study_name: str, study_dir: str):
        self.dataset = dataset
        self.dataset_train = None
        self.dataset_eval = None

        self.study_name = study_name
        self.study_dir = study_dir

        if not os.path.exists(self.study_dir):
            os.makedirs(self.study_dir)

        self.best_loss = np.inf

    def __call__(self, trial: optuna.Trial) -> float:
        try:
            # Model settings search space
            # attention_heads = trial.suggest_categorical("attention_heads", attention_heads_categorical)

            # Hyperparameter search space
            learning_rate = trial.suggest_float("learning_rate", lr_range[0], lr_range[1])
            # dataset_size = trial.suggest_categorical("dataset_size", dataset_size_categorical)
            # lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", lr_scheduler_types)
            # num_train_epochs = trial.suggest_int("num_train_epochs", train_epochs_range[0], train_epochs_range[1])
            # gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", gradient_accumulation_steps_categorical)
            # attn_dropout = trial.suggest_float("attn_dropout", attn_dropout_range[0], attn_dropout_range[1])
            # weight_decay = trial.suggest_float("weight_decay", weight_decay_range[0], weight_decay_range[1])
            # max_grad_norm = trial.suggest_float("max_grad_norm", max_grad_norm_range[0], max_grad_norm_range[1])

            # per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", per_device_train_batch_size_range[0], per_device_train_batch_size_range[1])
            # warmup_ratio = trial.suggest_float("warmup_ratio", warmup_ratio_range[0], warmup_ratio_range[1])

            # Reset the best loss
            self.best_loss = np.inf

            # Reset the perplexity
            self.best_perplexity = np.inf

            results_dir = f"{self.study_dir}/optuna_trial_{trial.number}"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # TrainingArguments setup
            self.training_args = TrainingArguments(
                output_dir=results_dir,
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
                evaluation_strategy="epoch",
                # eval_steps=0.2 / num_train_epochs - 0.001,
                save_strategy="no",
                # save_steps=0.5 / num_train_epochs - 0.001,
                logging_dir=f"{results_dir}/logs/",
                logging_strategy="steps",
                logging_steps=0.1 / num_train_epochs,
                report_to="none",
                # load_best_model_at_end=True,
                bf16=True,  # Enable mixed-precision training
                bf16_full_eval=True,  # Enable mixed-precision evaluation
                seed=seed,
            )

            # Prepare the dataset
            self.prepare_dataset(dataset_size, dataset_split)

            # Set the dropout rate and attention heads
            config_1B.attention_dropout = attn_dropout
            config_1B.num_attention_heads = attention_heads

            # Initialize the trainer
            trainer = SFTTrainer(
                model_init=self.model_init,
                args=self.training_args,
                train_dataset=self.dataset_train,
                eval_dataset=self.dataset_eval,
                dataset_text_field="text",
                packing=True,
                max_seq_length=context_length,
                tokenizer=tokenizer,
                callbacks=[self],
                # callbacks=[self, OptunaPruningCallback(trial, monitor="eval_loss")],
            )

            # Print the model size with suffix 'G' or 'M'
            model_size = sum(p.numel() for p in trainer.model.parameters())
            model_size = model_size / 1e9 if model_size > 1e9 else model_size / 1e6
            model_size_suffix = "G" if model_size > 1e3 else "M"

            dataset_train_size = len(self.dataset_train)
            dataset_eval_size = len(self.dataset_eval)

            # Print the hyperparameters
            print("Hyperparameters:")
            print(f"  Model size: {model_size:.2f}{model_size_suffix} parameters")
            print(f"  Hidden layers: {hidden_layers}")
            print(f"  Hidden size: {hidden_size}")
            print(f"  Intermediate size: {intermediate_size}")
            print(f"  Attention heads: {attention_heads}")
            print(f"  Attention dropout: {attn_dropout}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Learning rate scheduler type: {lr_scheduler_type}")
            print(f"  Epochs: {num_train_epochs}")
            print(f"  Warmup ratio: {warmup_ratio}")
            print(f"  Weight decay: {weight_decay}")
            print(f"  Max gradient norm: {max_grad_norm}")
            print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
            print(f"  Per device train batch size: {per_device_train_batch_size}")
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            print(f"  Device count: {device_count}")
            print(f"  Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps * device_count}")
            print(f"  Dataset size: {dataset_size}")
            print(f"  Training set size: {dataset_train_size}")
            print(f"  Evaluation set size: {dataset_eval_size}")

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
                        "dataset_size": dataset_size,
                        "dataset_split": dataset_split,
                        "stride": stride,
                        # Training settings
                        "seed": seed,
                        "lr_range": lr_range,
                        "lr_scheduler_type": lr_scheduler_type,
                        # "optim": optim,
                    },
                    f,
                )

            # Train the model
            trainer.train()
        except KeyboardInterrupt:
            print("\n\nTrial interrupted by user. Press Ctrl+C again to exit.")
            return self.best_loss

        # Save the model
        print(f"Saving the model to {results_dir}/model...")
        trainer.save_model(f"{results_dir}/model")

        # Return the best loss
        return self.best_loss

    def model_init(self) -> PreTrainedModel:
        self.model = MistralForCausalLM(config_1B).to(device)

        if self.training_args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        return self.model

    def prepare_dataset(self, dataset_size: int, dataset_split: float):
        if self.dataset_train is None or self.dataset_eval is None:
            prepared_dataset = None
            # Select the first dataset_size examples from the training set
            if dataset_size > 0:
                print("Selecting the first", dataset_size, "examples from the dataset...")
                prepared_dataset = self.dataset["train"].select(range(dataset_size))
            else:
                dataset_size = len(self.dataset["train"])
                print("Using the entire dataset of size", dataset_size)
                prepared_dataset = self.dataset["train"]

            # Split the dataset into training and evaluation sets (dataset_split% for training, 1-dataset_split% for evaluation)
            print("Splitting the dataset into training and evaluation sets...")
            print("Training set size:", round(dataset_size * dataset_split))
            print("Evaluation set size:", dataset_size - round(dataset_size * dataset_split))
            prepared_dataset = prepared_dataset.train_test_split(test_size=1-dataset_split, seed=seed)

            # Set the training and evaluation datasets
            self.dataset_train = prepared_dataset["train"]
            self.dataset_eval = prepared_dataset["test"]

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs["metrics"]["eval_loss"]
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss

        # Calculate the perplexity
        perplexity = math.exp(eval_loss)
        if perplexity < self.best_perplexity:
            self.best_perplexity = perplexity


# Optuna study
def run_optuna_study():
    storage_name = f"sqlite:///{study_dir}/optuna.db"

    if not os.path.exists(study_dir):
        os.makedirs(study_dir)

    # Use default sampler
    sampler = None

    # Use TPE sampler
    # sampler = optuna.samplers.TPESampler(seed=seed)

    # Use default pruner
    pruner = None

    # Use Median pruner
    # pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner

    )
    objective = Objective(dataset, study_name, study_dir)

    try:
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\n\nOptuna study interrupted by user")

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

    # Save the study
    study_file = f"{study_dir}/optuna_study.pkl"
    print(f"Saving the study to {study_file}...")
    with open(study_file, "wb") as f:
        pickle.dump(study, f)


# Start the hyperparameter search
if __name__ == "__main__":
    run_optuna_study()
