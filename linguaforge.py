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

parser.add_argument("--hidden_size", type=int, default=2048, help="Size of the hidden states in the transformer layers")
parser.add_argument("--intermediate_size", type=int, default=4096,
                    help="Size of the feed-forward network in the transformer layers")
parser.add_argument("--num_attention_heads", type=int, default=32, help="Number of attention heads")
parser.add_argument("--num_key_value_heads", type=int, default=8, help="Number of key-value heads")
parser.add_argument("--num_hidden_layers", type=int, default=1, help="Number of transformer layers")
parser.add_argument("--context_length", type=int, default=1024, help="Maximum sequence length")
parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention")
parser.add_argument("--liger_kernels", action="store_true", help="Use LIGER kernels to increase throughput and reduce memory usage")

# Add the arguments for the dataset settings
parser.add_argument("--dataset_name_or_path", type=str, default=None, required=True, help="Name of the dataset to use")
parser.add_argument("--dataset_config", type=str, default=None, help="Configuration of the dataset to use")
parser.add_argument("--dataset_train_split_name", type=str, default="train", help="Name of the training split")
parser.add_argument("--dataset_test_split_name", type=str, default=None, help="Name of the test split")
parser.add_argument("--reformat_dataset", type=str, default=None, help="Reformat the dataset using the specified script. The script must contain a 'format_example' function that takes a batch of examples and returns a batch of formatted examples. Example: ```\npython def format_example(batch):\n    if 'text' in batch:\n        batch['text'] = [text.lower() for text in batch['text']]\n    return batch\n```")
parser.add_argument("--dataset_size_train", type=int, default=0, help="Number of examples to use for the training set. Default is to use whatever is left after the splits.")
parser.add_argument("--dataset_size_val", type=int, default=None, help="Number of examples to use for the validation set used during training. Keep it small so in-training evals don't take too long")
parser.add_argument("--dataset_size_test", type=int, default=None, help="Number of examples to use for the test set used after training. Should be a decent size.")
parser.add_argument("--dataset_split", type=float, default=0.9, help="Percentage of examples to use for training. Must be less than 1.0")
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
parser.add_argument("--opt_hidden_size", action="store_true", help="Optimize the hidden size")
parser.add_argument("--hidden_size_categorical", type=int, nargs="+",
                    default=[128, 256, 512, 1024], help="Categorical values for the hidden size")
parser.add_argument("--opt_intermediate_size", action="store_true", help="Optimize the intermediate size")
parser.add_argument("--intermediate_size_categorical", type=int, nargs="+",
                    default=[256, 512, 1024, 2048, 4096], help="Categorical values for the intermediate size")
parser.add_argument("--opt_attention", action="store_true", help="Optimize the number of attention heads and kv heads simultaneously (recommended)")
parser.add_argument("--opt_num_attention_heads", action="store_true", help="Optimize the number of attention heads")
parser.add_argument("--num_attention_heads_categorical", type=int, nargs="+",
                    default=[8, 12, 16, 32], help="Categorical values for the number of attention heads")
parser.add_argument("--opt_num_key_value_heads", action="store_true", help="Optimize the number of key-value heads")
parser.add_argument("--num_key_value_heads_categorical", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32], help="Categorical values for the number of key-value heads")
parser.add_argument("--opt_num_hidden_layers", action="store_true", help="Optimize the number of hidden layers")
parser.add_argument("--num_hidden_layers_range", type=int, nargs=2,
                    default=[1, 18], help="Range of hidden layers to use for hyperparameter search")
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

import itertools
import json
import math
import random
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import optuna
from datasets import (Dataset, DatasetDict, IterableDataset,
                      IterableDatasetDict, load_dataset, load_from_disk)
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          EarlyStoppingCallback, PreTrainedModel)
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
study_dir = f"{args.output_dir}/optuna-study-{timestamp}"

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
def split_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
    split_sizes: Dict[str, Union[int, float]],
    split_priority: Optional[List[str]] = None,
    shuffle: bool = False,
    seed: int = 42,
    subset_strategy: str = 'head',
    allow_reshuffle: bool = False,
    batch_size: int = 1000,
    num_proc: Optional[int] = None
) -> Union[DatasetDict, IterableDatasetDict]:
    """
    Split or subset a dataset into multiple splits based on given sizes.

    Args:
        dataset: The input dataset.
        split_sizes: A dictionary of split names and their sizes (int or float).
        split_priority: Order in which to process splits. Defaults to keys of split_sizes.
        shuffle: Whether to shuffle the dataset before splitting.
        seed: Random seed for shuffling.
        subset_strategy: Strategy for subsetting existing splits ('head', 'tail', 'random').
        allow_reshuffle: Whether to allow reshuffling of existing splits.
        batch_size: Batch size for processing iterable datasets.
        num_proc: Number of processes to use for processing.

    Returns:
        A DatasetDict or IterableDatasetDict containing the requested splits.
    """
    # Convert to appropriate dictionary type if necessary
    if isinstance(dataset, (Dataset, IterableDataset)):
        new_dataset = DatasetDict() if isinstance(dataset, Dataset) else IterableDatasetDict()
        new_dataset["train"] = dataset
        dataset = new_dataset
        del new_dataset

    is_iterable = isinstance(dataset, IterableDatasetDict)
    
    # Validate and process split sizes
    total_size = sum(len(split) for split in dataset.values()) if not is_iterable else None
    split_sizes = _process_split_sizes(split_sizes, total_size)
    
    # Set split priority
    split_priority = split_priority or list(split_sizes.keys())
    
    # Initialize result
    result = IterableDatasetDict() if is_iterable else DatasetDict()
    
    # Process each split
    remaining = None
    for split_name in split_priority:
        size = split_sizes[split_name]
        
        if split_name in dataset and len(dataset[split_name]) >= size:
            # Use existing split
            result[split_name] = _subset_split(dataset[split_name], size, subset_strategy, seed, allow_reshuffle)
        else:
            # Create new split from remaining data
            if remaining is None:
                remaining = dataset.get('train', next(iter(dataset.values())))
                if shuffle:
                    remaining = remaining.shuffle(seed=seed)
            
            if is_iterable:
                result[split_name], remaining = _split_iterable(remaining, size, batch_size, seed)
            else:
                result[split_name], remaining = _split_dataset(remaining, size)
        
    if remaining is not None:
        result["train"] = _subset_split(remaining, split_sizes['train'], subset_strategy, seed, allow_reshuffle)

    return result

def _process_split_sizes(split_sizes: Dict[str, Union[int, float]], total_size: Optional[int]) -> Dict[str, int]:
    """Convert split sizes to integers and validate."""
    if all(isinstance(size, float) for size in split_sizes.values()):
        assert sum(split_sizes.values()) <= 1, "Float sizes must sum to <= 1"
        assert total_size is not None, "Total size must be known for fractional splitting"
        return {name: math.floor(total_size * size) for name, size in split_sizes.items()}
    elif all(isinstance(size, int) for size in split_sizes.values()):
        return split_sizes
    else:
        raise ValueError("All split sizes must be either int or float")

def _subset_split(split: Union[Dataset, IterableDataset], size: int, strategy: str, seed: int, allow_reshuffle: bool) -> Union[Dataset, IterableDataset]:
    """Subset an existing split based on the given strategy."""
    if isinstance(split, IterableDataset):
        return split.take(size)
    
    if size == len(split) or size == 0:
        return split
    
    if strategy == 'head':
        return split.select(range(size))
    elif strategy == 'tail':
        return split.select(range(len(split) - size, len(split)))
    elif strategy == 'random':
        if allow_reshuffle:
            return split.shuffle(seed=seed).select(range(size))
        else:
            indices = list(range(len(split)))
            rng = random.Random(seed)
            rng.shuffle(indices)
            return split.select(indices[:size])
    else:
        raise ValueError(f"Unknown subset strategy: {strategy}")

def _split_dataset(dataset: Dataset, size: int) -> Tuple[Dataset, Dataset]:
    """Split a Dataset into two parts."""
    return dataset.select(range(size)), dataset.select(range(size, len(dataset)))

def _split_iterable(dataset: IterableDataset, size: int, batch_size: int, seed: int) -> Tuple[IterableDataset, IterableDataset]:
    """Split an IterableDataset into two parts."""
    return dataset.take(size), dataset.skip(size)

def reformat_dataset(dataset, reformat_dataset):
    processing_module = load_processing_script(reformat_dataset)
    if hasattr(processing_module, 'format_example'):
        dataset.set_transform(processing_module.format_example)
    else:
        raise AttributeError("The processing script must contain a 'format_example' function")


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


def generate_valid_attention_kv_configs() -> List[Dict[str, int]]:
    attention_heads = args.num_attention_heads_categorical if args.opt_num_attention_heads else [args.num_attention_heads]
    key_value_heads = args.num_key_value_heads_categorical if args.opt_num_key_value_heads else [args.num_key_value_heads]
    
    valid_configs = []
    
    for ah, kvh in itertools.product(attention_heads, key_value_heads):
        if ah % kvh == 0:
            valid_configs.append(f"{ah}_{kvh}")
    
    return valid_configs


if args.opt_attention or (args.opt_num_attention_heads and args.opt_num_key_value_heads):
    valid_attention_kv_configs = generate_valid_attention_kv_configs()


def parse_attention_kv_config(config_str: str) -> Dict[str, int]:
    num_attention_heads, num_key_value_heads = map(int, config_str.split('_'))
    return {
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads
    }


# Hyperparameter search space
def hp_space(trial: optuna.Trial) -> dict:
    global args
    
    space = {}

    # Training hyperparameters
    if args.opt_lr:
        space["learning_rate"] = trial.suggest_float("learning_rate", args.lr_range[0], args.lr_range[1], log=True)
    if args.opt_lr_scheduler_type:
        space["lr_scheduler_type"] = trial.suggest_categorical("lr_scheduler_type", args.lr_scheduler_types)
    if args.opt_train_epochs:
        space["num_train_epochs"] = trial.suggest_int("num_train_epochs", args.train_epochs_range[0], args.train_epochs_range[1])
    if args.opt_per_device_train_batch_size:
        space["per_device_train_batch_size"] = trial.suggest_int(
            "per_device_train_batch_size", args.per_device_train_batch_size_range[0], args.per_device_train_batch_size_range[1])
    if args.opt_gradient_accumulation_steps:
        space["gradient_accumulation_steps"] = trial.suggest_categorical(
            "gradient_accumulation_steps", args.gradient_accumulation_steps_categorical)
    if args.opt_weight_decay:
        space["weight_decay"] = trial.suggest_float("weight_decay", args.weight_decay_range[0], args.weight_decay_range[1], log=True)
    if args.opt_max_grad_norm:
        space["max_grad_norm"] = trial.suggest_float("max_grad_norm", args.max_grad_norm_range[0], args.max_grad_norm_range[1], log=True)
    if args.opt_warmup_ratio:
        space["warmup_ratio"] = trial.suggest_uniform("warmup_ratio", args.warmup_ratio_range[0], args.warmup_ratio_range[1])
    if args.opt_warmup_steps:
        space["warmup_steps"] = trial.suggest_int("warmup_steps", args.warmup_steps_range[0], args.warmup_steps_range[1])
    if args.opt_grokfast_ema_alpha:
        space["grokfast_ema_alpha"] = trial.suggest_float("grokfast_ema_alpha", args.grokfast_ema_alpha_range[0], args.grokfast_ema_alpha_range[1])
    if args.opt_grokfast_ema_lambda:
        space["grokfast_ema_lambda"] = trial.suggest_float("grokfast_ema_lambda", args.grokfast_ema_lambda_range[0], args.grokfast_ema_lambda_range[1])

    # Model Configuration
    if args.opt_dtype:
        space["dtype"] = trial.suggest_categorical("dtype", args.dtype_categorical)
    if args.opt_hidden_size:
        space["hidden_size"] = trial.suggest_categorical("hidden_size", args.hidden_size_categorical)
    if args.opt_intermediate_size:
        space["intermediate_size"] = trial.suggest_categorical("intermediate_size", args.intermediate_size_categorical)
    if args.opt_num_hidden_layers:
        space["num_hidden_layers"] = trial.suggest_int("num_hidden_layers", args.num_hidden_layers_range[0], args.num_hidden_layers_range[1])

    if args.opt_attention or (args.opt_num_attention_heads and args.opt_num_key_value_heads):
        suggestion = trial.suggest_categorical("attention_kv_config", valid_attention_kv_configs)
        suggestion = parse_attention_kv_config(suggestion)
        space.update(suggestion)
    else:
        if args.opt_num_attention_heads:
            space["num_attention_heads"] = trial.suggest_categorical("num_attention_heads", args.num_attention_heads_categorical)
        elif args.opt_num_key_value_heads:
            space["num_key_value_heads"] = trial.suggest_categorical("num_key_value_heads", args.num_key_value_heads_categorical)

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
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_position_embeddings=args.context_length,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=tokenizer.pad_token_id if tokenizer is not None else None,
            sliding_window=None,
            torch_dtype=args.dtype,
        )

        if args.flash_attn:
            model_config.update({"attn_implementation":"flash_attention_2"})

        # If this is a trial, set the config from the trial
        if trial is not None:
            space = trial.params

            if "dtype" in space:
                model_config["torch_dtype"] = space["dtype"]
            if "hidden_size" in space:
                model_config["hidden_size"] = space["hidden_size"]
            if "intermediate_size" in space:
                model_config["intermediate_size"] = space["intermediate_size"]
            if "num_hidden_layers" in space:
                model_config["num_hidden_layers"] = space["num_hidden_layers"]
            if "num_attention_heads" in space:
                model_config["num_attention_heads"] = space["num_attention_heads"]
            if "num_key_value_heads" in space:
                model_config["num_key_value_heads"] = space["num_key_value_heads"]
            if "attention_kv_config" in space:
                model_config.update(parse_attention_kv_config(space["attention_kv_config"]))

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
    dataset = load_from_disk(args.dataset_name_or_path, keep_in_memory=args.keep_dataset_in_memory)
except:
    dataset = load_dataset(args.dataset_name_or_path, args.dataset_config, keep_in_memory=args.keep_dataset_in_memory, streaming=args.dataset_streaming)

if dataset is None:
    raise ValueError(
        f"Could not load dataset from {args.dataset_name_or_path} ({args.dataset_config})."
    )

# Prepare the dataset
if "val" in dataset:
    dataset["validation"] = dataset["val"]
    del dataset["val"]

split_sizes={}
split_priority=[]
if args.dataset_size_test is not None:
    split_sizes["test"] = args.dataset_size_test
    split_priority.append("test")
if args.dataset_size_val is not None:
    split_sizes["validation"] = args.dataset_size_val
    split_priority.append("validation")
if args.dataset_size_train is not None:
    split_sizes["train"] = args.dataset_size_train
    split_priority.append("train")

dataset = split_dataset(
    dataset=dataset,
    split_sizes=split_sizes,
    split_priority=split_priority,
    shuffle=args.shuffle,
    seed=args.dataset_shuffle_seed if args.dataset_shuffle_seed is not None else args.seed
)

# Apply reformatting if specified
if args.reformat_dataset:
    reformat_dataset(dataset, args)

# Save the prepared dataset
if args.save_prepared_dataset or args.save_prepared_dataset_only:
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
                "dataset_split_sizes": split_sizes,
                "dataset_shuffle_seed": args.dataset_shuffle_seed if args.dataset_shuffle_seed is not None else args.seed,
                "dataset_reformat": args.reformat_dataset,
                "dataset_save_path": args.dataset_save_path,
                "shuffle": args.shuffle,
                "batch_size": args.dataset_batch_size if args.dataset_batch_size is not None else args.batch_size,
                "stride": args.stride,
                "num_cpus": args.num_cpus,
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
    eval_dataset=dataset["validation"],
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

    # Evaluation
    eval_results = None
    if "test" in dataset:
        print_if_main_process("Evaluating the model on the heldout test dataset...")
        # Tokenize the test dataset
        def tokenize_function(examples):
            global tokenizer
            tokenized_data = tokenizer(
                examples["text"],  # Adjust this to match your dataset's column name
                padding="max_length",
                truncation=True,
                max_length=args.context_length,
                return_tensors="pt"
            )

            tokenized_data["labels"] = tokenized_data["input_ids"].clone()
            return tokenized_data

        # Tokenize the test dataset
        tokenized_test_dataset = dataset["test"].map(
            tokenize_function,
            batched=True,
            batch_size=args.dataset_batch_size if args.dataset_batch_size is not None else args.batch_size,
            remove_columns=dataset["test"].column_names
        )
        eval_results = trainer.evaluate(tokenized_test_dataset)

    # Display the results
    print_if_main_process(f"Final results:")
    print_if_main_process(f"Train Loss: {trainer.state.log_history[-2]['train_loss']:.4f}")
    print_if_main_process(f"Train PPL: {math.exp(trainer.state.log_history[-2]['train_loss']):.4f}")
    # print_if_main_process(f"Validation Loss: {trainer.state.log_history[-3]['eval_loss']:.4f}")
    # print_if_main_process(f"Validation PPL: {math.exp(trainer.state.log_history[-3]['eval_loss']):.4f}")
    if eval_results:
        print_if_main_process(f"Evaluation Loss: {eval_results['eval_loss']:.4f}")
        print_if_main_process(f"Evaluation PPL: {math.exp(eval_results['eval_loss']):.4f}")
    print_if_main_process()
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
    global args

    study_db_path = f"{results_dir}/optuna.db"
    study_storage = f"sqlite:///{study_db_path}"

    optuna_kwargs = {
        "study_name": args.study_name,
        "storage": study_storage,
        "gc_after_trial": True,
    }

    # Set up the pruner
    class CustomPruner(optuna.pruners.BasePruner):
        def __init__(self, n_warmup_steps: int=500):
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
        n_trials=args.n_trials,
        direction="minimize" if not args.greater_is_better else "maximize",
        backend="optuna",
        pruner=pruner,
        **optuna_kwargs,
    )

    if is_main_process:
        # Load the study from the database
        study = optuna.load_study(study_name=args.study_name, storage=study_storage)

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
