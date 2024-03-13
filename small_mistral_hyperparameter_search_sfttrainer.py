from datasets import load_dataset, DatasetDict
import bitsandbytes as bnb
import json
import math
import numpy as np
import optuna
import os
import pickle
import torch
from torch import nn
from transformers import (
    MistralForCausalLM,
    MistralConfig,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from transformers.trainer_pt_utils import get_parameter_names
from trl import set_seed, SFTTrainer
from typing import Union
from transformers import PreTrainedModel

hf_token = "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"  # I'm a bad person for hardcoding this

# Use Mistral-7B-v0.1 as a template for the model settings
template_model_name = "mistralai/Mistral-7B-v0.1"

# Model settings - Model size: 760.26M parameters
hidden_layers = 10  # Number of transformer layers
hidden_size = 2048  # Size of the hidden states in the transformer layers
intermediate_size = 8192  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
attn_dropout = 0.1  # Dropout rate for the attention probabilities
context_length = 2048  # Maximum sequence length

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "/media/gronkomatic/Embiggen/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = 1000  # Number of examples to use from the dataset
dataset_split = 0.9  # Percentage of examples to use for training
stride = 50  # Stride for splitting the input into multiple sequences. Doesn't work with Mistral according to CoPilot, but what would they know?

# Training settings
seed = 42  # Random seed for reproducibility
learning_rate = 9.8e-5  # Learning rate for the AdamW optimizer
lr_scheduler_type = "linear"  # Use a cosine annealing learning rate scheduler
num_train_epochs = 1  # Number of training epochs
per_device_train_batch_size = 1  # Batch size per GPU/TPU core/CPU for training
warmup_ratio = 0.15  # Ratio of the number of warmup steps to the total number of training steps
weight_decay = 0.01  # Weight decay for the AdamW optimizer
max_grad_norm = 1.0  # Maximum gradient norm
gradient_accumulation_steps = 1  # Number of steps to accumulate gradients for
gradient_checkpointing = False  # Causes a segfault when enabled
optim = "adamw_torch"  # Use PyTorch's AdamW optimizer

# Optuna study settings
study_name = "mistral-small_hyperparameter_search-attention_heads-8-32-1000"
study_dir = f"./results/{study_name}"
n_trials = 4  # Number of hyperparameter search trials
dataset_size_range = [500, 1000]  # Range of dataset sizes to use for hyperparameter search
lr_range = [1e-5, 1e-4]  # Range of learning rates to use for hyperparameter search
lr_scheduler_types = ["linear", "cosine", "cosine_with_restarts"]  # Learning rate scheduler types
attention_heads_categorical = [8, 16, 24, 32]  # Categorical values for the number of attention heads
train_epochs_range = [1, 10]  # Range of training epochs to use for hyperparameter search
per_device_train_batch_size_range = [1, 3]  # Range of batch sizes to use for hyperparameter search
warmup_ratio_range = [0.1, 0.2]  # Range of warmup ratios to use for hyperparameter search
gradient_accumulation_steps_range = [1, 2]  # Range of gradient accumulation steps to use for hyperparameter search
attn_dropout_range = [0.0, 0.1]  # Range of attention dropout rates to use for hyperparameter search


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

        if not os.path.exists(self.study_dir):
            os.makedirs(self.study_dir)

        self.best_loss = np.inf

    def __call__(self, trial: optuna.Trial) -> float:
        # Model settings search space
        attention_heads = trial.suggest_categorical("attention_heads", attention_heads_categorical)

        # Hyperparameter search space
        # learning_rate = trial.suggest_float("learning_rate", lr_range[0], lr_range[1])
        # lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", lr_scheduler_types)
        # num_train_epochs = trial.suggest_int("num_train_epochs", train_epochs_range[0], train_epochs_range[1])
        # per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", per_device_train_batch_size_range[0], per_device_train_batch_size_range[1])
        # warmup_ratio = trial.suggest_float("warmup_ratio", warmup_ratio_range[0], warmup_ratio_range[1])
        # gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", gradient_accumulation_steps_range[0], gradient_accumulation_steps_range[1])
        # attn_dropout = trial.suggest_float("attn_dropout", attn_dropout_range[0], attn_dropout_range[1])
        # dataset_size = trial.suggest_int("dataset_size", dataset_size_range[0], dataset_size_range[1])

        # Reset the best loss
        self.best_loss = np.inf

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
            evaluation_strategy="epoch",
            eval_steps=0.5 / num_train_epochs,
            logging_dir=f"{results_dir}/logs/",
            logging_strategy="no",
            logging_steps=0.5 / num_train_epochs,
            report_to="none",
            save_strategy="no",
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
        trainer = CustomSFTTrainer(
            model_init=self.model_init,
            args=self.training_args,
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
        print(f"  Attention heads: {attention_heads}")
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
    objective = Objective(dataset, study_name, study_dir)
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
    with open(f"{study_name}/optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)


# Start the hyperparameter search
if __name__ == "__main__":
    run_optuna_study()
