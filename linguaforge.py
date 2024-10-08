import math
import os
import sys
from typing import Dict

# Set environment variables for NCCL blocking wait and error handling
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Enable parallelism in the tokenizers library
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Print the local rank
is_main_process = os.environ.get("LOCAL_RANK", 0) == "0" or os.environ.get("LOCAL_RANK", -1) == -1
print("Main process:", is_main_process)
print("Local rank:", os.environ.get("LOCAL_RANK", -1))

import argparse


def int_or_float(value):
    try:
        float_value = float(value)
        if float_value.is_integer():
            return int(float_value)
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}")



# Custom action to parse key-value pairs
class KeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for item in values:
            key, value = item.split("=")
            if isinstance(value, str):
                try:
                    if "." in value or "e" in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = value


# Set up command-line arguments with argparse
parser = argparse.ArgumentParser(description="Train a model using the SFTTrainer")

parser.add_argument("--project_name", type=str, required=True, help="Name of the project")
parser.add_argument(
    "--output_dir", type=str, default=f"./results", help="Directory to save the results"
)
parser.add_argument("--run_name", type=str, default=None, help="Name of the run")

# Add the arguments for the distributed training settings
parser.add_argument("--num_cpus", type=int, default=None, help="Number of CPUs to use")
parser.add_argument("--gpu_devices", type=str, default="0", help="GPU devices to use")

# Add the arguments for the model settings
model_name_group = parser.add_mutually_exclusive_group(required=True)
model_name_group.add_argument("--pretrained_model_name_or_path", type=str, help="Name or path of the pretrained model")
model_name_group.add_argument("--template_model_name", type=str, help="Template model name")
model_name_group.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume training from")

parser.add_argument("--hidden_layers", type=int, default=1, help="Number of transformer layers")
parser.add_argument("--hidden_size", type=int, default=2048, help="Size of the hidden states in the transformer layers")
parser.add_argument("--intermediate_size", type=int, default=4096,
                    help="Size of the feed-forward network in the transformer layers")
parser.add_argument("--attention_heads", type=int, default=32, help="Number of attention heads")
parser.add_argument("--num_key_value_heads", type=int, default=8, help="Number of key-value heads")
parser.add_argument("--context_length", type=int, default=1024, help="Maximum sequence length")
parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention")
parser.add_argument("--liger_kernels", action="store_true", help="Use LIGER kernels to increase throughput and reduce memory usage")

# Add the arguments for the dataset settings
parser.add_argument("--dataset_name_or_path", type=str, default=None, required=True, help="Name of the dataset to use")
parser.add_argument("--dataset_config", type=str, default="default", help="Configuration of the dataset to use")
parser.add_argument("--dataset_train_split_name", type=str, default="train", help="Name of the training split")
parser.add_argument("--dataset_test_split_name", type=str, default=None, help="Name of the test split")
parser.add_argument("--reformat_dataset", type=str, default=None, help="Reformat the dataset using the specified script. The script must contain a 'format_example' function that takes a batch of examples and returns a batch of formatted examples. Example: ```\npython def format_example(batch):\n    if 'text' in batch:\n        batch['text'] = [text.lower() for text in batch['text']]\n    return batch\n```")
parser.add_argument("--dataset_size_train", type=int, default=0, help="Number of examples to use from the training set. Set to 0 to use the entire training set")
parser.add_argument("--dataset_size_test", type=int, default=0, help="Number of examples to use from the test set. Set to 0 to use the entire test set")
parser.add_argument("--dataset_split", type=float, default=0.9, help="Percentage of examples to use for training if < 1, or number of examples if >= 1")
parser.add_argument("--stride", type=int, default=150, help="Stride for splitting the input into multiple sequences")
parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
parser.add_argument("--keep_dataset_in_memory", action="store_true", help="Keep the dataset in memory")
parser.add_argument("--dataset_streaming", action="store_true", help="Enable dataset streaming")
parser.add_argument("--dataset_batch_size", type=int, default=1000, help="Batch size for processing the dataset")
parser.add_argument("--dataset_packing", action="store_true", help="Enable dataset packing")
parser.add_argument("--save_prepared_dataset", action="store_true", help="Save the prepared dataset")
parser.add_argument("--save_prepared_dataset_only", action="store_true", help="Save the prepared dataset and exit")
parser.add_argument("--dataset_save_path", type=str, default="dataset", help="Path to save the prepared dataset")

# Add the arguments for the tokenization settings
parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Name or path of the tokeniser")
parser.add_argument("--additional_special_tokens", type=str, nargs="+",
                    default=None, help="Additional special tokens to add to the tokenizer")
parser.add_argument("--chat_template", type=str, default=None, help="Chat template for chatbot training")

# Add the arguments for the training settings
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--dataset_shuffle_seed", type=int, default=42, help="Random seed for dataset shuffling only")
parser.add_argument("--dtype", type=str, default="bfloat16",
                    help="Data type to use for the model",
                    choices=["float16", "bfloat16", "float32"])
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the AdamW optimizer")
parser.add_argument(
    "--lr_scheduler_type",
    type=str,
    default="reduce_lr_on_plateau",
    choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
        "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau",
        "cosine_with_min_lr", "warmup_stable_decay"
    ],
    help="Learning rate scheduler type",
)
parser.add_argument("--lr_scheduler_args", nargs="+", action=KeyValueAction, help="Arguments for the learning rate scheduler")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--num_train_steps", type=int, default=-1, help="Number of training steps. Supercedes num_train_epochs")
parser.add_argument("--logging_steps", type=int_or_float, default=None, help="Number of steps between logging")
parser.add_argument("--include_num_input_tokens_seen", action="store_true", help="Include the number of input tokens seen in the log output")
parser.add_argument("--eval_steps", type=int_or_float, default=None, help="Number of steps between evaluations")
parser.add_argument("--eval_on_start", action="store_true", help="Evaluate the model at the start of training")
parser.add_argument("--save_steps", type=int_or_float, default=None, help="Number of steps between saving the model")
parser.add_argument("--save_total_limit", type=int, default=None, help="Number of checkpoints to keep")
parser.add_argument("--load_best_model_at_end", action="store_true", help="Load the best model at the end of training")
parser.add_argument("--metric_for_best_model", type=str, default=None, help="Metric to use for the best model")
parser.add_argument("--greater_is_better", action="store_true", help="The metric for the best model is greater when true")
parser.add_argument("--evals_per_epoch", type=int, default=1, help="Number of evaluations per epoch")
parser.add_argument("--auto_find_batch_size", action="store_true", help="Automatically find the batch size")
parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                    help="Batch size per GPU/TPU core/CPU for training")
parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                    help="Batch size per GPU/TPU core/CPU for evaluation")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of steps to accumulate gradients for")
parser.add_argument("--eval_accumulation_steps", type=int, default=None,
                    help="Number of steps to accumulate evaluation results for before moving to CPU. Saves VRAM during eval.")
parser.add_argument("--warmup_ratio", type=float, default=0.0,
                    help="Ratio of the number of warmup steps to the total number of training steps")
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the AdamW optimizer")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
parser.add_argument(
    "--optimizer",
    type=str,
    default="adamw_bnb_8bit",
    choices=[
        'adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla',
        'adamw_torch_npu_fused', 'adamw_apex_fused', 'adafactor', 'adamw_anyprecision',
        'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit', 'lion_32bit',
        'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_lion_32bit', 'paged_lion_8bit',
        'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit', 'rmsprop_bnb_32bit', 'galore_adamw',
        'galore_adamw_8bit', 'galore_adafactor', 'galore_adamw_layerwise',
        'galore_adamw_8bit_layerwise', 'galore_adafactor_layerwise'
    ],
    help="Optimizer to use"
)
parser.add_argument("--optimizer_args", nargs="+", action=KeyValueAction, help="Arguments for the optimizer")
parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile")

# Logging settings
parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases")

# Early stopping
parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs to wait before early stopping")
parser.add_argument("--early_stopping_threshold", type=float, default=0.0, help="Minimum change in the monitored quantity to qualify as an improvement")

# Grokfast Accelerated Grokking
parser.add_argument("--grokfast_ema", action="store_true", help="Enable Grokfast EMA slow-gradient amplification")
parser.add_argument("--grokfast_ema_alpha", type=float, default=0.98, help="Alpha parameter for Grokfast EMA")
parser.add_argument("--grokfast_ema_lambda", type=float, default=2.0, help="Lambda parameter for Grokfast EMA")

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
                    default=[8, 16, 32], help="Categorical values for the number of attention heads")
parser.add_argument("--opt_num_key_value_heads", action="store_true", help="Optimize the number of key-value heads")
parser.add_argument("--num_key_value_heads_categorical", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32], help="Categorical values for the number of key-value heads")
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
                    default=[0.01, 0.1], help="Range of warmup ratios to use for hyperparameter search")
parser.add_argument("--opt_warmup_steps", action="store_true", help="Optimize the warmup steps")
parser.add_argument("--warmup_steps_range", type=int, nargs=2,
                    default=[0, 1000], help="Range of warmup steps to use for hyperparameter search")
parser.add_argument("--opt_hidden_layers", action="store_true", help="Optimize the number of hidden layers")
parser.add_argument("--hidden_layers_range", type=int, nargs=2,
                    default=[1, 18], help="Range of hidden layers to use for hyperparameter search")
parser.add_argument("--opt_grokfast_ema_alpha", action="store_true", help="Optimize Grokfast EMA alpha")
parser.add_argument("--grokfast_ema_alpha_range", type=float, nargs=2,
                    default=[0.8, 0.99], help="Range of Grokfast alpha values to use for hyperparameter search")
parser.add_argument("--opt_grokfast_ema_lambda", action="store_true", help="Optimize Grokfast EMA lambda")
parser.add_argument("--grokfast_ema_lambda_range", type=float, nargs=2,
                    default=[0.5, 5], help="Range of Grokfast lambda values to use for hyperparameter search")

args = parser.parse_args()

# Set the number of CPUs
if args.num_cpus is None:
    args.num_cpus = 1

# Set devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("No CUDA device found. Please use a CUDA-enabled device for training.")


import json
import time
import warnings

import evaluate
import optuna
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    EarlyStoppingCallback,
    PreTrainedModel,
)
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.trainer_utils import EvalPrediction
from transformers.utils import PaddingStrategy, logging
from trl import SFTConfig, SFTTrainer, set_seed

# Ignore the warning about gathering scalars
warnings.filterwarnings(
    "ignore",
    "Was asked to gather along dimension 0, but all "
    "input tensors were scalars; will instead unsqueeze "
    "and return a vector.",
    append=True,
)

# Ignore the FutureWarning about passing arguments to Accelerator
warnings.filterwarnings(
    "ignore",
    "Passing the following arguments to `Accelerator` "
    "is deprecated and will be removed in version 1.0 of Accelerate:",
    category=FutureWarning,
    append=True,
)


# Function that prints to the console only if the process is the main process
def print_if_main_process(*args, **kwargs):
    global is_main_process
    if is_main_process:
        print(*args, **kwargs)


# Get the logger
logger = logging.get_logger(__name__)

# Set the logging level to ERROR
logger.setLevel(logging.ERROR)

# Disable the logging if not the main process
if not is_main_process:
    logger.setLevel(logging.ERROR)

timestamp = time.strftime("%Y%m%d-%H%M%S")

# Directory to save the results
if args.resume_from_checkpoint:
    results_dir = args.output_dir  # Results path is passed on the command line when resuming from a checkpoint
else:
    args.output_dir = args.output_dir + "/" + args.project_name
    if args.run_name:
        results_dir = f"{args.output_dir}/{args.run_name}-{timestamp}"
    else:
        results_dir = f"{args.output_dir}/run-{timestamp}"

# Training settings
# Set dtype to the appropriate torch dtype
args.dtype = (
    torch.bfloat16 if args.dtype == "bfloat16"
    else torch.float16 if args.dtype == "float16"
    else torch.float32
)

# Optuna study settings
study_name = args.study_name  # Name of the Optuna study
study_dir = f"{args.output_dir}/optuna-study-{timestamp}"
n_trials = args.n_trials  # Number of hyperparameter search trials
lr_range = args.lr_range  # Range of learning rates to use for hyperparameter search
dtype_categorical = args.dtype_categorical  # Categorical values for the data type to use
# Categorical values for the learning rate scheduler type
lr_scheduler_types = args.lr_scheduler_types
attention_heads_categorical = args.attention_heads_categorical  # Categorical values for the number of attention heads
train_epochs_range = args.train_epochs_range  # Range of training epochs to use for hyperparameter search
warmup_ratio_range = args.warmup_ratio_range  # Range of warmup ratios to use for hyperparameter search
warmup_steps_range = args.warmup_steps_range  # Range of warmup steps to use for hyperparameter search
# Range of batch sizes to use for hyperparameter search
per_device_train_batch_size_range = args.per_device_train_batch_size_range
# Categorical values for the number of gradient accumulation steps
gradient_accumulation_steps_categorical = args.gradient_accumulation_steps_categorical
weight_decay_range = args.weight_decay_range  # Range of weight decay values to use for hyperparameter search
max_grad_norm_range = args.max_grad_norm_range  # Range of maximum gradient norms to use for hyperparameter search
hidden_layers_range = args.hidden_layers_range  # Range of hidden layers to use for hyperparameter search

# Set the final output directory
if args.run_hyperparameter_search:
    results_dir = study_dir

# Set seed for reproducibility
set_seed(args.seed)

# Create the results directory
if is_main_process and not os.path.exists(results_dir):
    os.makedirs(results_dir)

print(f"Using device: {device}")


def load_processing_script(script_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dataset_processing", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Prepare the dataset
def prepare_dataset(
    dataset: DatasetDict,
    dataset_split: float,
    dataset_size_train: int,
    dataset_size_test: int,
    shuffle: bool = False,
    batch_size: int = 1000,
) -> DatasetDict:
    """
    Prepare the dataset for training and evaluation by splitting it into training and evaluation sets, and selecting a subset of examples from the training and evaluation sets.
    TODO: It would be better if dataset transforms were used to prepare the dataset. This would allow for more flexibility in the dataset preparation process. Refactor this function to use dataset transforms.
    """
    global args

    seed = args.dataset_shuffle_seed if args.dataset_shuffle_seed is not None else args.seed

    print_if_main_process("Preparing the dataset...")
    prepared_dataset = None

    # If the dataset is already split into train and test and/or validate, use it as is. Prefer "validation" split over "test" split.
    if "validation" in dataset:
        dataset["test"] = dataset["validation"]
        del dataset["validation"]
    if "test" in dataset:
        if shuffle:
            dataset["train"] = dataset["train"].shuffle(seed)
            dataset["test"] = dataset["test"].shuffle(seed)
        if dataset_size_train > 0:
            print_if_main_process("Selecting", dataset_size_train, "examples from the training set...")
            dataset["train"] = dataset["train"].select(range(dataset_size_train))
        if dataset_size_test > 0:
            print_if_main_process("Selecting", dataset_size_test, "examples from the test set...")
            dataset["test"] = dataset["test"].select(range(dataset_size_test))
        prepared_dataset = dataset
    else:
        if shuffle:
            dataset["train"] = dataset["train"].shuffle(seed)

        # Select the first dataset_size examples from the training set
        if dataset_size_train > 0:
            print_if_main_process("Selecting", dataset_size_train, "examples from the dataset...")
            prepared_dataset = dataset["train"].select(range(dataset_size_train))
        else:
            dataset_size_train = len(dataset["train"])
            print_if_main_process("Using the entire dataset of size", dataset_size_train)
            prepared_dataset = dataset["train"]

        # Split the dataset into training and evaluation sets (dataset_split% for training, 1-dataset_split% for evaluation)
        print_if_main_process("Splitting the dataset into training and evaluation sets...")
        print_if_main_process("Training set size:", round(len(prepared_dataset) * dataset_split))
        print_if_main_process("Evaluation set size:", len(prepared_dataset) - round(len(prepared_dataset) * dataset_split))
        prepared_dataset = prepared_dataset.train_test_split(train_size=dataset_split, seed=seed, shuffle=shuffle)

    if args.reformat_dataset:
        processing_module = load_processing_script(args.reformat_dataset)
        if hasattr(processing_module, 'format_example'):
            prepared_dataset.set_transform(processing_module.format_example)
        else:
            raise AttributeError("The processing script must contain a 'format_example' function")

    # Return the training and evaluation datasets
    return prepared_dataset


# Load the evaluation metrics
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")
# metric_rouge = evaluate.load("rouge")
# metric_bleu = evaluate.load("bleu")


# Compute the evaluation metrics
def compute_metrics(eval_pred: EvalPrediction, compute_result=False):
    with torch.no_grad():
        # Get the logits, attention mask, and labels
        logits = eval_pred.predictions.detach()
        metric_labels = eval_pred.label_ids.detach()
        attention_mask = eval_pred.inputs["attention_mask"].detach()

        # Shift the labels and attention mask to the left
        metric_labels = metric_labels[..., 1:]
        attention_mask = attention_mask[..., 1:]
        logits = logits[..., :-1, :]

        predictions = torch.argmax(logits, dim=-1)

        # Mask out the padding tokens
        if attention_mask is None:
            predictions = predictions * attention_mask
            metric_labels = metric_labels * attention_mask

        # Flatten the input and move to CPU
        metric_labels = metric_labels.flatten().cpu()
        predictions = predictions.flatten().cpu()
        attention_mask = attention_mask.flatten().cpu()

        metric_accuracy.add_batch(predictions=predictions, references=metric_labels)
        metric_f1.add_batch(predictions=predictions, references=metric_labels)

        del logits, metric_labels, predictions, attention_mask
        torch.cuda.empty_cache()

    if compute_result:
        return {
            "accuracy": metric_accuracy.compute()["accuracy"],
            "f1": metric_f1.compute(average="micro")["f1"],
        }
    else:
        return {}


# Hyperparameter search objective function
def compute_objective(metrics: Dict[str, float]) -> float:
    """
    The objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metric is provided in args.metric_for_best_model. Otherwise, it is the metric provided in args.metric_for_best_model.

    Args:
        metrics (`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        `float`: The objective to minimize or maximize
    """
    if args.metric_for_best_model:
        # Does it have "eval_" prefix?
        if args.metric_for_best_model.startswith("eval_"):
            metric = args.metric_for_best_model
        elif "eval_" + args.metric_for_best_model in metrics:
            metric = "eval_" + args.metric_for_best_model
        else:
            metric = args.metric_for_best_model
        return metrics[metric]
    else:
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
        space["num_attention_heads"] = trial.suggest_categorical("num_attention_heads", attention_heads_categorical)
    if args.opt_num_key_value_heads:
        space["num_key_value_heads"] = trial.suggest_categorical("num_key_value_heads", args.num_key_value_heads_categorical)
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
    if args.opt_warmup_steps:
        space["warmup_steps"] = trial.suggest_int("warmup_steps", warmup_steps_range[0], warmup_steps_range[1])
    if args.opt_hidden_layers:
        space["hidden_layers"] = trial.suggest_int("hidden_layers", hidden_layers_range[0], hidden_layers_range[1])
    if args.opt_grokfast_ema_alpha:
        space["grokfast_ema_alpha"] = trial.suggest_float("grokfast_ema_alpha", args.grokfast_ema_alpha_range[0], args.grokfast_ema_alpha_range[1])
    if args.opt_grokfast_ema_lambda:
        space["grokfast_ema_lambda"] = trial.suggest_float("grokfast_ema_lambda", args.grokfast_ema_lambda_range[0], args.grokfast_ema_lambda_range[1])

    return space


# Set up the tokenizer
def tokenizer_init(model_name_or_path: str) -> AutoTokenizer:
    global args

    # Load tokenizer
    print_if_main_process(f"Loading the tokenizer from {model_name_or_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = args.context_length

    # Set up truncation and padding
    tokenizer.set_truncation_and_padding(
        padding_strategy=PaddingStrategy.LONGEST,
        truncation_strategy=TruncationStrategy.LONGEST_FIRST,
        max_length=args.context_length,
        stride=args.stride,
        pad_to_multiple_of=8,
        padding_side="right"
    )

    # Add special tokens to the tokenizer
    additional_special_tokens = [
        # "<|im_start|>", "<|im_end|>",
        # "<|named_user|>",  # Named user. For future use. Example: "<|im_start|><|named_user|>Alice\n<Alice's message><|im_end|>"
        # "<|named_assistant|>",  # Named assistant. For future use. Example: "<|im_start|><|named_assistant|>Assistant George\n<Assistant George's message><|im_end|>"
        # "<|mem_start|>", "<|mem_end|>",  # Memory start and end tokens. For future use. Store hidden information in the context, e.g. "<|mem_start|>Alice's birthday is 12th May.<|mem_end|>"
        # "<|pause|>",  # Pause token. For future use. See https://arxiv.org/abs/2310.02226.pdf Think before you speak: Training Language Models With Pause Tokens
    ]

    # Add additional special tokens from args
    if args.additional_special_tokens:
        additional_special_tokens += args.additional_special_tokens

    # Add <|spare_1|>, <|spare_2|>, etc. to the tokenizer to make the vocab size a multiple of 8
    if (len(additional_special_tokens) + len(tokenizer)) % 8 != 0:
        for i in range(1, 8 - (len(tokenizer) + len(additional_special_tokens)) % 8 + 1):
            additional_special_tokens.append(f"<|spare_{i}|>")
            print_if_main_process(f"Added <|spare_{i}|> to the tokenizer.")

    # Add the special tokens to the tokenizer
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens},
        replace_additional_special_tokens=False,
    )

    if len(additional_special_tokens) > 0 and is_main_process:
        print(f"Additional special tokens added to the tokenizer.")

        # Print the token IDs of the special tokens
        for token in additional_special_tokens:
            print(f"{token}: {tokenizer(token)}")

    # Assert that the vocab size is a multiple of 8
    assert (
        len(tokenizer)
    ) % 8 == 0, "The vocabulary size is not a multiple of 8. Fix the padding code!"

    # Set up the chat template
    if args.chat_template:
        tokenizer.chat_template = args.chat_template

    return tokenizer


# Initialize the model
def model_init(trial: optuna.Trial) -> PreTrainedModel:
    global tokenizer
    if trial is not None:
        print_if_main_process("\033[93m" + f"Trial {trial.number}" + "\033[0m")
        # Print the hyperparameters as a single-line JSON string
        print_if_main_process("\033[93m" + json.dumps(trial.params) + "\033[0m")

    print_if_main_process("Initialising the model...")

    if args.dtype == torch.float16:
        print_if_main_process("Can't train in float16. Switching to float32")
        args.dtype = torch.float32

    model_kwargs = {"torch_dtype":args.dtype}
    if args.flash_attn:
        model_kwargs.update({"attn_implementation":"flash_attention_2"})

    # If a pretrained model is provided, load it
    if args.pretrained_model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_name_or_path,
            **model_kwargs,
        )
    else:
        # Set the initial model configuration
        model_config = dict(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.hidden_layers,
            num_attention_heads=args.attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_position_embeddings=args.context_length,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=tokenizer.pad_token_id if tokenizer is not None else None,
            sliding_window=None,
            torch_dtype=args.dtype,
        )

        if args.flash_attn:
            model_config.update({"attn_implementation":"flash_attention_2"})

        # If this is a trial, set the hyperparameters from the trial
        if trial is not None:
            space = trial.params

            if "dtype" in space:
                model_config["torch_dtype"] = space["dtype"]
            if "num_attention_heads" in space:
                model_config["num_attention_heads"] = space["num_attention_heads"]
            if "num_key_value_heads" in space:
                model_config["num_key_value_heads"] = space["num_key_value_heads"]
            if "hidden_layers" in space:
                model_config["num_hidden_layers"] = space["hidden_layers"]
            if "hidden_size" in space:
                model_config["hidden_size"] = space["hidden_size"]

        model_config = AutoConfig.from_pretrained(args.template_model_name, **model_config)
        model = AutoModelForCausalLM.from_config(model_config)

        # Move the model to the device
        model = model.to(device)

    # Resize the token embeddings to match the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Print the number of model parameter with suffix 'B' for billion or 'M' for million
    model_size = sum(p.numel() for p in model.parameters())
    model_size_suffix = "B" if model_size > 1e9 else "M"
    model_size = model_size / 1e9 if model_size > 1e9 else model_size / 1e6

    print_if_main_process(f"Model size: {model_size:.2f}{model_size_suffix}")

    return model


def save_model(path: str) -> str:
    model_path = f"{path}/model"
    trainer.save_model(model_path)
    # print(f"Model saved to {model_path}")

    return model_path


# Save the command-line arguments to a file
if is_main_process:
    with open(f"{results_dir}/command_line_args.txt", "w") as f:
        f.write(" ".join(sys.argv[1:]))

# Load the dataset
print_if_main_process(
    f"Loading the dataset from {args.dataset_name_or_path} ({args.dataset_config})..."
)
dataset = None
try:
    dataset = load_dataset(args.dataset_name_or_path, args.dataset_config, keep_in_memory=args.keep_dataset_in_memory, streaming=args.dataset_streaming)
except ValueError as ve:
    if "Please use `load_from_disk` instead." in str(ve):
        dataset = load_from_disk(args.dataset_name_or_path, keep_in_memory=args.keep_dataset_in_memory)

if dataset is None:
    raise ValueError(
        f"Could not load dataset from {args.dataset_name_or_path} ({args.dataset_config})."
    )
elif "train" not in dataset:
    new_dataset = DatasetDict()
    new_dataset["train"] = dataset
    dataset = new_dataset
    del new_dataset

# Prepare the dataset
dataset = prepare_dataset(
    dataset=dataset,
    dataset_split=args.dataset_split,
    dataset_size_train=args.dataset_size_train,
    dataset_size_test=args.dataset_size_test,
    shuffle=args.shuffle,
    batch_size=args.dataset_batch_size,
)

# Save the prepared dataset
if args.save_prepared_dataset:
    # Only save if main process
    if is_main_process:
        dataset.save_to_disk(args.dataset_save_path)
    print_if_main_process(f"Prepared dataset saved to {args.dataset_save_path}")
    if args.save_prepared_dataset_only:
        print_if_main_process("Exiting...")
        exit()

# Save the dataset configuration
if is_main_process:
    with open(f"{results_dir}/dataset_config.json", "w") as f:
        json.dump(
            {
                "dataset_name_or_path": args.dataset_name_or_path,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "dataset_size": len(dataset),
                "shuffle": args.shuffle,
                "batch_size": args.dataset_batch_size,
                "stride": args.stride,
            },
            f,
            indent=2,
        )

if args.wandb:
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = args.project_name

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "false"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

# TrainingArguments setup
training_kwargs = {}

# Set the output directory
training_kwargs.update({"output_dir": results_dir})

if args.eval_steps:
    training_kwargs.update({
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps
    })
elif args.evals_per_epoch:
    if args.evals_per_epoch == 1:
        training_kwargs.update({"eval_strategy": "epoch"})
    else:
        training_kwargs.update({
            "eval_strategy": "steps",
            "eval_steps": (1 / args.evals_per_epoch) / args.num_train_epochs
        })
else:
    training_kwargs.update({"eval_strategy": "epoch"})

print_if_main_process(f"eval_strategy = {training_kwargs["eval_strategy"]}")
if training_kwargs["eval_strategy"] == "steps":
    print_if_main_process(f"eval_steps = {training_kwargs["eval_steps"]}")

if args.save_steps is None:
    training_kwargs.update({"save_strategy": "no"})
elif args.save_steps == 0:
    training_kwargs.update({"save_strategy": "epoch"})
else:
    training_kwargs.update({"save_strategy": "steps", "save_steps": args.save_steps})
    
print_if_main_process(f"save_strategy = {training_kwargs["save_strategy"]}")
if training_kwargs["save_strategy"] == "steps":
    print_if_main_process(f"save_steps = {training_kwargs["save_steps"]}")

if args.logging_steps is None:
    training_kwargs.update({"logging_strategy": "no"})
elif args.logging_steps == 0:
    training_kwargs.update({"logging_strategy": "epoch"})
else:
    training_kwargs.update({
        "logging_strategy": "steps",
        "logging_steps": training_kwargs["eval_steps"] if args.logging_steps is None else args.logging_steps,
    })

print_if_main_process(f"logging_strategy = {training_kwargs["logging_strategy"]}")
if training_kwargs["logging_strategy"] == "steps":
    print_if_main_process(f"logging_steps = {training_kwargs["logging_steps"]}")

# Set the run name
if args.run_name:
    training_kwargs.update({"run_name": f"{args.run_name}-{timestamp}"})
elif args.resume_from_checkpoint and args.wandb:
    print_if_main_process("When resuming from a checkpoint and logging to Weights & Biases, a run name must be provided with the --run_name argument.")
    exit()
else:
    training_kwargs.update({"run_name": f"run-{timestamp}"})

# Add the GrokFast options if they're passed
if args.grokfast_ema:
    training_kwargs.update({
        "grokfast_ema": args.grokfast_ema,
        "grokfast_ema_alpha": args.grokfast_ema_alpha,
        "grokfast_ema_lambda": args.grokfast_ema_lambda
    })

# Enable liger kernels
if args.liger_kernels:
    training_kwargs.update({"use_liger": True})

# Enable `torch.compile()` once support is better
if args.torch_compile:
    print("torch.compile() is not yet supported. Skipping.")
    # training_kwargs.update({"torch_compile": True})

sfttrainer_args = {}
tokenizer = None

if args.resume_from_checkpoint:
    sfttrainer_args["model"] = f"{results_dir}/{args.resume_from_checkpoint}"
else:
    sfttrainer_args["model_init"] = model_init
    # Tokenizer setup
    if args.pretrained_model_name_or_path:
        # Load the tokenizer from the pretrained model
        tokenizer = tokenizer_init(args.pretrained_model_name_or_path)
        sfttrainer_args.update({"tokenizer": tokenizer})
    elif args.template_model_name:
        # Load the tokenizer from the template model
        tokenizer = tokenizer_init(args.template_model_name)
        sfttrainer_args.update({"tokenizer": tokenizer})

if args.tokenizer_name_or_path:
    sfttrainer_args["tokenizer"] = tokenizer_init(args.tokenizer_name_or_path)

training_args = SFTConfig(
    logging_dir=f"{results_dir}/logs/",
    num_train_epochs=args.num_train_epochs,
    max_steps=args.num_train_steps,
    auto_find_batch_size=args.auto_find_batch_size,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio,
    warmup_steps=args.warmup_steps,
    save_total_limit=args.save_total_limit,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    lr_scheduler_kwargs=args.lr_scheduler_args,
    optim=args.optimizer,
    weight_decay=args.weight_decay,
    seed=args.seed,
    data_seed=args.dataset_shuffle_seed if args.dataset_shuffle_seed else args.seed,
    bf16=(args.dtype == torch.bfloat16),
    bf16_full_eval=(args.dtype == torch.bfloat16),
    fp16=(args.dtype == torch.float16),
    fp16_full_eval=(args.dtype == torch.float16),
    report_to="wandb" if args.wandb else "none",
    remove_unused_columns=True,
    load_best_model_at_end=args.load_best_model_at_end,
    metric_for_best_model=args.metric_for_best_model,
    dataset_text_field="text",
    dataset_batch_size=args.dataset_batch_size,
    packing=args.dataset_packing,
    max_seq_length=args.context_length,
    dataset_num_proc=args.num_cpus,
    dataloader_num_workers=args.num_cpus,
    accelerator_config={"split_batches": True},
    ddp_find_unused_parameters=False,
    batch_eval_metrics=True,
    include_num_input_tokens_seen=True,
    eval_on_start=args.eval_on_start,
    include_inputs_for_metrics=True,
    **training_kwargs
)

# Initialize the trainer
trainer = SFTTrainer(
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    **sfttrainer_args
)

# If early stopping is enabled, add the callback
if args.early_stopping:
    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
    )


def run_training():
    # Print the hyperparameters
    print_if_main_process("Hyperparameters:")
    print_if_main_process(f"  Learning rate: {args.learning_rate}")
    print_if_main_process(f"  Learning rate scheduler: {args.lr_scheduler_type}")
    print_if_main_process(f"  Per-device train batch size: {args.per_device_train_batch_size}")
    print_if_main_process(f"  Epochs: {args.num_train_epochs}")
    if args.warmup_steps > 0:
        print_if_main_process(f"  Warmup steps: {args.warmup_steps}")
    else:
        print_if_main_process(f"  Warmup ratio: {args.warmup_ratio}")
    print_if_main_process(f"  Attention heads: {args.attention_heads}")
    print_if_main_process(
        f"  Gradient accumulation steps: {args.gradient_accumulation_steps}"
    )
    print_if_main_process(f"  Weight decay: {args.weight_decay}")
    print_if_main_process(f"  Results directory: {results_dir}")
    print_if_main_process(f"  Optimizer: {args.optimizer}")
    print_if_main_process()

    # Save the hyperparameters to a file
    hyperparameters = {
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio if args.warmup_steps == 0 else 0,
        "warmup_steps": args.warmup_steps,
        "attention_heads": args.attention_heads,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "results_dir": results_dir,
        "optim": args.optimizer,
    }
    with open(f"{results_dir}/hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)

    # Train the model
    try:
        trainer.train(resume_from_checkpoint=bool(args.resume_from_checkpoint))
    except KeyboardInterrupt:
        # Save the training progress if main process
        if is_main_process:
            print("\nSaving the training progress...")
            save_model(results_dir)
            print("Training progress saved.")
            print("Training interrupted by user.")
            print(f"Resume training by running the following command:\npython {" ".join(sys.argv[1:])} --output_dir {results_dir} --resume_from_checkpoint")
        else:
            time.sleep(5)
        exit()

    print_if_main_process("Training complete!")
    print_if_main_process()

    # Save the model
    model_path = save_model(results_dir)

    # Display the results
    print_if_main_process("Results directory:", results_dir)
    print_if_main_process("Model saved to:", model_path)
    print_if_main_process("Hyperparameters saved to:", f"{results_dir}/hyperparameters.json")
    print_if_main_process("Logs saved to:", f"{results_dir}/logs/")
    print_if_main_process()
    print_if_main_process("You can now fine-tune the model further or use it for generating text.")


def run_study():
    """
    Run a hyperparameter optimization study using Optuna.

    This function sets up the study, runs the hyperparameter search, loads the study results,
    visualizes the study results, and saves the best run.
    """
    study_db_path = f"{results_dir}/optuna.db"
    study_storage = f"sqlite:///{study_db_path}"

    optuna_kwargs = {
        "study_name": study_name,
        "storage": study_storage,
        "gc_after_trial": True,
    }

    # Set up the pruner
    class CustomPruner(optuna.pruners.BasePruner):
        def __init__(self, n_warmup_steps=500):
            self.n_warmup_steps = n_warmup_steps
            self.median_pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)

        def prune(self, study, trial):
            step = trial.last_step

            if step < self.n_warmup_steps:
                return False

            value = trial.intermediate_values[step]
            if value is None:
                return False
            # Check for NaN or zero
            if math.isnan(value) or value == 0.0:
                return True

            return self.median_pruner.prune(study, trial)

    # Create an instance of the custom pruner
    pruner = CustomPruner(n_warmup_steps=args.warmup_steps)

    # Run the hyperparameter search
    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,
        compute_objective=compute_objective,
        n_trials=n_trials,
        direction="minimize" if not args.greater_is_better else "maximize",
        backend="optuna",
        pruner=pruner,
        **optuna_kwargs,
    )

    if is_main_process:
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
        optuna.visualization.plot_edf(study).write_html(f"{vis_dir}/edf.html")

        # Save the best run
        best_run_path = f"{results_dir}/best_run.json"
        with open(best_run_path, "w") as f:
            json.dump(best_run, f, indent=4)
        print_if_main_process(f"Best run saved to {best_run_path}")

        # Save the best model
        best_model_path = save_model(results_dir)

        # Print the best run
        print_if_main_process("Best run:")
        print_if_main_process(json.dumps(best_run, indent=4))

        # Display the results
        print_if_main_process("Results directory:", results_dir)
        print_if_main_process("Best run saved to:", best_run_path)
        print_if_main_process("Study saved to:", study_db_path)
        print_if_main_process("Visualizations saved to:", vis_dir)
        print_if_main_process("Best model saved to:", best_model_path)
        print_if_main_process("Logs saved to:", f"{results_dir}/logs/")
        print_if_main_process()
        print_if_main_process("To view the training logs, run the following command:")
        print_if_main_process(f"tensorboard --logdir {results_dir}/logs/")


# Run whatever is needed
if args.run_hyperparameter_search:
    run_study()
else:
    run_training()
