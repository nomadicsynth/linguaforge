import numpy as np
import optuna
import pickle
import torch
from datasets import load_dataset, DatasetDict
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer, TrainingArguments
from transformers import TrainerCallback
from trl import set_seed, SFTTrainer
from typing import Union

# Model settings
hidden_layers = 25  # Number of transformer layers
hidden_size = 1024  # Size of the hidden states in the transformer layers
intermediate_size = 2048  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
context_length = 1024  # Maximum sequence length
stride = 50  # Stride for splitting the input into multiple sequences
tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"  # Name of the tokenizer to use

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "D:/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = 1000  # Number of examples to use from the dataset. 0 means all examples
dataset_split = 0.9  # Percentage of examples to use for training

# Training settings
seed = 42
lr_scheduler_type = "linear"
optim = "adamw_torch"  # Use PyTorch's AdamW optimizer
n_trials = 50  # Number of hyperparameter search trials

# Set seed for reproducibility
set_seed(seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration for a hypothetical <1B parameter model
config_1B = LlamaConfig(
    vocab_size=32000,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=hidden_layers,
    num_attention_heads=attention_heads,
    max_position_embeddings=context_length,
    pad_token_id=2,
    torch_dtype="bfloat16",
    # attn_implementation="flash_attention_2",  # Disable torch_dtype="bfloat16" if using flash_attention_2
)

# Load tokenizer
print(f"Loading the tokenizer from {tokenizer_name}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token

# Prepare dataset
print(f"Loading the dataset from {dataset_name} ({dataset_config})...")
dataset = load_dataset(dataset_path, dataset_config)

# Select the first dataset_size examples from the training set
if dataset_size > 0:
    print("Selecting the first", dataset_size, "examples from the dataset...")
    dataset = dataset["train"].select(range(dataset_size))
else:
    dataset_size = len(dataset["train"])
    print("Using the entire dataset of size", dataset_size)
    dataset = dataset["train"]

# Shuffling the dataset
print("Shuffling the dataset...")
dataset = dataset.shuffle(seed=seed)

# Split the dataset into training and evaluation sets (dataset_split% for training, 1-dataset_split% for evaluation)
print("Splitting the dataset into training and evaluation sets...")
print("Training set size:", int(dataset_size * dataset_split))
print("Evaluation set size:", dataset_size - int(dataset_size * dataset_split))
dataset = dataset.train_test_split(test_size=1-dataset_split, seed=seed)


# Objective function for Optuna
class Objective(TrainerCallback):
    def __init__(self, dataset: Union[dict, DatasetDict]):
        self.dataset_train = dataset["train"]
        self.dataset_eval = dataset["test"]
        self.best_loss = np.inf

    def __call__(self, trial: optuna.Trial) -> float:
        # Hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4)
        num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
        per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 1, 3)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.5)
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 32)

        # Reset the best loss
        self.best_loss = np.inf

        # Define the model initialization function
        def model_init():
            return LlamaForCausalLM(config_1B).to(device)

        # TrainingArguments setup
        training_args = TrainingArguments(
            output_dir=f"./results/optuna_trial_{trial.number}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            logging_dir=f"./logs/optuna_trial_{trial.number}",
            logging_strategy="epoch",
            report_to="none",  # Avoid clutter
            optim=optim,
            save_strategy="epoch",
            bf16=True,  # Enable mixed-precision training
            bf16_full_eval=True,  # Enable mixed-precision evaluation
            seed=seed,
        )

        # Initialize the trainer
        trainer = SFTTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_eval,
            dataset_text_field="text",
            max_seq_length=context_length,
            tokenizer=tokenizer,
            callbacks=[self],
        )

        # Train the model
        trainer.train()
        return self.best_loss

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs["metrics"]["eval_loss"]
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss


# Optuna study
def run_optuna_study():
    study = optuna.create_study(direction="minimize")
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
