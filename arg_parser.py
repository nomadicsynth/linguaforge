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


def setup_arg_parser():
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
    parser.add_argument("--dataset_cache_path", type=str, default=None, help="Path to save the prepared dataset cache")

    # Add the arguments for the tokenization settings
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Name or path of the tokeniser")
    parser.add_argument("--additional_special_tokens", type=str, nargs="+",
                        default=None, help="Additional special tokens to add to the tokenizer")
    parser.add_argument("--chat_template", type=str, default=None, help="Chat template for chatbot training")
    parser.add_argument("--stride", type=int, default=150, help="Stride for splitting the input into multiple sequences")

    # Add the arguments for the training settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_init_seed", type=int, default=None, help="Random seed for model initialization only")
    parser.add_argument("--dataset_shuffle_seed", type=int, default=None, help="Random seed for dataset shuffling only")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type to use for the model",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the model")
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
            'adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_torch_npu_fused',
            'adamw_apex_fused', 'adafactor', 'adamw_anyprecision', 'adamw_torch_4bit', 'ademamix',
            'sgd', 'adagrad', 'adamw_bnb_8bit', 'adamw_8bit', 'ademamix_8bit', 'lion_8bit', 'lion_32bit',
            'paged_adamw_32bit', 'paged_adamw_8bit', 'paged_ademamix_32bit', 'paged_ademamix_8bit',
            'paged_lion_32bit', 'paged_lion_8bit', 'rmsprop', 'rmsprop_bnb', 'rmsprop_bnb_8bit',
            'rmsprop_bnb_32bit', 'galore_adamw', 'galore_adamw_8bit', 'galore_adafactor',
            'galore_adamw_layerwise', 'galore_adamw_8bit_layerwise', 'galore_adafactor_layerwise',
            'lomo', 'adalomo', 'grokadamw', 'schedule_free_adamw', 'schedule_free_sgd'
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

    return parser


def parse_args(args=None, namespace=None):
    parser = setup_arg_parser()
    return parser.parse_args(args, namespace)


# You can test the parser directly if this file is run as a script
if __name__ == "__main__":
    parse_args(["--help"])
