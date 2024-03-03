import numpy as np
import optuna
import pickle
import time
import torch
from CustomLlamaModel import CustomLlamaModel
from datasets import load_dataset
from transformers import LlamaConfig, AutoTokenizer, TrainingArguments
from transformers import TrainerCallback
from trl import set_seed, SFTTrainer

# Model settings
hidden_layers = 12  # Number of transformer layers
hidden_size = 1024  # Size of the hidden states in the transformer layers
intermediate_size = 2048  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
context_length = 2048  # Length of the input context
stride = 50  # Stride for splitting the input into multiple sequences

# Dataset settings
dataset_name = "wikimedia/wikipedia"  # Name of the dataset to use
dataset_config = "20231101.en"  # Configuration of the dataset to use
dataset_path = "D:/ai-stuff/datasets/wikipedia"  # Path to the dataset
dataset_size = 100  # Number of examples to use from the dataset. 0 means all examples
dataset_split = 0.9  # Percentage of examples to use for training

# Training settings
seed = 42
epochs = 5  # Number of training epochs
batch_size = 3  # Number of sequences to process in parallel
gradient_accumulation_steps = 4  # Number of update steps to accumulate before performing a backward pass
# warmup_steps = 100 / gradient_accumulation_steps  # Number of warmup steps for the learning rate scheduler
warmup_steps = 10  # Number of warmup steps for the learning rate scheduler

run = "2"
output_dir = "./results/run-" + run
logging_dir = output_dir + "/logs"
# final_dir = "./final"

learning_rate = 5e-5
lr_scheduler_type = "linear"
optim = "adamw_torch"  # Use PyTorch's AdamW optimizer

logging_strategy = "epoch"
logging_steps = 10  # Log training loss every X steps

evaluation_strategy = "epoch"
eval_steps = 1 / (epochs * 4)  # Evaluate every 25% of an epoch
save_strategy = "steps"
save_steps = eval_steps

load_best_model_at_end = True
metric_for_best_model = "loss"

# Write the configuration to a JSON file
training_config = {
    "hidden_layers": hidden_layers,
    "hidden_size": hidden_size,
    "intermediate_size": intermediate_size,
    "attention_heads": attention_heads,
    "context_length": context_length,
    "stride": stride,
    "seed": seed,
    "epochs": epochs,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "logging_strategy": logging_strategy,
    "logging_steps": logging_steps,
    "warmup_steps": warmup_steps,
    "learning_rate": learning_rate,
    "lr_scheduler_type": lr_scheduler_type,
    "optim": optim,
    "evaluation_strategy": evaluation_strategy,
    "eval_steps": eval_steps,
    "save_strategy": save_strategy,
    "save_steps": save_steps,
    "load_best_model_at_end": load_best_model_at_end,
    "metric_for_best_model": metric_for_best_model,
    "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# Set seed for reproducibility
set_seed(seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration for a hypothetical 1B parameter model
config_1B = LlamaConfig(
    vocab_size=32000,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=hidden_layers,
    num_attention_heads=attention_heads,
    max_position_embeddings=context_length,
    pad_token_id=2,
    torch_dtype="bfloat16",
    # attn_implementation="flash_attention_2",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token

# Prepare dataset
print("Loading the dataset...")
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
    def __init__(self, dataset):
        self.dataset_train = dataset["train"]
        self.dataset_eval = dataset["test"]
        self.best_loss = np.inf

    def __call__(self, trial):
        # Hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4)
        num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
        # per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
        warmup_steps = trial.suggest_int("warmup_steps", 10, 100)
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 8)

        # Reset the best loss
        self.best_loss = np.inf

        # Define the model initialization function
        def model_init():
            return CustomLlamaModel(config_1B).to(device)

        # TrainingArguments setup
        training_args = TrainingArguments(
            output_dir=f"./results/optuna_trial_{trial.number}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            evaluation_strategy=evaluation_strategy,
            logging_dir=f"./logs/optuna_trial_{trial.number}",
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            report_to="none",  # Avoid clutter
            optim=optim,
            save_strategy="no",
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
    study.optimize(objective, n_trials=10)  # Adjust the number of trials as needed

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

    # Optionally, save the study
    with open(f"./results/optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)


# Start the hyperparameter search
if __name__ == "__main__":
    run_optuna_study()
