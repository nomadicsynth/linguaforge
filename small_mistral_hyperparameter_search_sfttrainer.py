from datasets import load_dataset, DatasetDict
import json
import math
import numpy as np
import os
import optuna
import pickle
import torch
from transformers import (
    MistralForCausalLM,
    MistralConfig,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from trl import set_seed, SFTTrainer
from typing import Union

hf_token = "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"

# Model settings
hidden_layers = 12  # Number of transformer layers
hidden_size = 2048  # Size of the hidden states in the transformer layers
intermediate_size = 4096  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
attn_dropout = 0.1  # Dropout rate for the attention probabilities
context_length = 2048  # Maximum sequence length
template_model_name = "mistralai/Mistral-7B-v0.1"  # Name of the tokenizer to use

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size_range = [500, 500]  # Range of dataset sizes to use for hyperparameter search
dataset_split = 0.9  # Percentage of examples to use for training
stride = 50  # Stride for splitting the input into multiple sequences

# Training settings
seed = 42
lr_range = [1e-5, 1e-4]  # Range of learning rates to use for hyperparameter search
lr_scheduler_types = ["linear", "cosine", "cosine_with_restarts"]  # Learning rate scheduler types
optim = "adamw_torch"  # Use PyTorch's AdamW optimizer
n_trials = 20  # Number of hyperparameter search trials

# Set seed for reproducibility
set_seed(seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
tokenizer.stride = stride

# Load the dataset
print(f"Loading the dataset from {dataset_name} ({dataset_config})...")
dataset = load_dataset(dataset_path, dataset_config)


# Custom callback for Optuna pruning
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial, monitor: str):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Retrieve the metric to monitor
        metric_value = metrics.get(self.monitor)
        if metric_value is None:
            raise ValueError(f"The monitored metric '{self.monitor}' was not found.")

        # Report the current metric value to Optuna and check for pruning
        self.trial.report(metric_value, step=state.epoch)
        if self.trial.should_prune() or math.isnan(metric_value) or math.isinf(metric_value):
            message = f"Trial was pruned at epoch {state.epoch}."
            raise optuna.exceptions.TrialPruned(message)


# Objective function for Optuna
class Objective(TrainerCallback):
    def __init__(self, dataset: Union[dict, DatasetDict], study_name: str, study_dir: str):
        self.dataset = dataset
        self.dataset_train = None
        self.dataset_eval = None

        self.study_name = study_name
        self.study_dir = study_dir

        self.best_loss = np.inf

    def __call__(self, trial: optuna.Trial) -> float:
        # Model settings search space
        attention_heads = trial.suggest_categorical("attention_heads", [8, 16, 24, 32])
        # Hyperparameter search space
        # learning_rate = trial.suggest_float("learning_rate", lr_range[0], lr_range[1])
        learning_rate = 9.8e-5
        # lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", lr_scheduler_types)
        lr_scheduler_type = "linear"
        # num_train_epochs = trial.suggest_int("num_train_epochs", 1, 10)
        num_train_epochs = 5
        # per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 1, 3)
        per_device_train_batch_size = 1
        # warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.2)
        warmup_ratio = 0.15
        # gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 2)
        gradient_accumulation_steps = 1
        # attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.1)
        # dataset_size = trial.suggest_int("dataset_size", dataset_size_range[0], dataset_size_range[1])
        dataset_size = 1000

        # Reset the best loss
        self.best_loss = np.inf

        # Define the model initialization function
        def model_init():
            return MistralForCausalLM(config_1B).to(device)

        results_dir = f"{self.study_dir}/optuna_trial_{trial.number}"
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
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            evaluation_strategy="epoch",
            eval_steps=0.5 / num_train_epochs,
            logging_dir=f"{results_dir}/logs/",
            logging_strategy="no",
            logging_steps=0.5 / num_train_epochs,
            report_to="none",
            optim=optim,
            save_strategy="no",
            bf16=True,  # Enable mixed-precision training
            bf16_full_eval=True,  # Enable mixed-precision evaluation
            seed=seed,
        )

        # Prepare the dataset
        self.prepare_dataset(dataset_size, dataset_split)

        # Set the dropout rate
        config_1B.attention_dropout = attn_dropout

        # Initialize the trainer
        trainer = SFTTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_eval,
            dataset_text_field="text",
            packing=True,
            max_seq_length=context_length,
            tokenizer=tokenizer,
            callbacks=[self, OptunaPruningCallback(trial, monitor="eval_loss")],
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
        print(f"  Learning rate: {learning_rate}")
        print(f"  Learning rate scheduler type: {lr_scheduler_type}")
        print(f"  Epochs: {num_train_epochs}")
        print(f"  Warmup ratio: {warmup_ratio}")
        print(f"  Attention dropout: {attn_dropout}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Dataset train size: {dataset_train_size}")
        print(f"  Dataset eval size: {dataset_eval_size}")


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
                    "dataset_size_range": dataset_size_range,
                    "dataset_split": dataset_split,
                    "stride": stride,
                    # Training settings
                    "seed": seed,
                    "lr_range": lr_range,
                    "lr_scheduler_types": lr_scheduler_types,
                    "optim": optim,
                },
                f,
            )

        # Train the model
        trainer.train()

        # Return the best loss
        return self.best_loss

    def prepare_dataset(self, dataset_size: int, dataset_split: float):
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


# Optuna study
def run_optuna_study():
    results_dir = "./results"
    study_name = "mistral-small_hyperparameter_search-attention_heads-8-32-1000"
    study_dir = f"{results_dir}/{study_name}"
    storage_name = f"sqlite:///{study_dir}/optuna.db"

    if not os.path.exists(study_dir):
        os.makedirs(study_dir)

    # Use TPE sampler
    sampler = optuna.samplers.TPESampler(seed=seed)

    # Use Median pruner
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner

    )
    objective = Objective(dataset)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

    # Save the study
    with open(f"./results/optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)


# Start the hyperparameter search
if __name__ == "__main__":
    run_optuna_study()
