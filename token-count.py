import argparse
from transformers import AutoTokenizer, PaddingStrategy, TruncationStrategy
from datasets import load_dataset, DatasetDict

parser = argparse.ArgumentParser(description="Counts the number of tokens in a dataset.")
parser.add_argument("--model_name", type=str, default="gpt2", help="Model to load the tokenizer from.")
parser.add_argument("--sequence_length", type=int, default=1024, help="Maximum sequence length.")
parser.add_argument("--stride", type=int, default=64, help="Stride for splitting the input into multiple sequences.")
parser.add_argument("--additional_special_tokens", type=str, nargs="+", default=[], help="Additional special tokens to add to the tokenizer.")
parser.add_argument("--chat_template", type=str, default=None, help="Chat template.")
parser.add_argument("--dataset_path", type=str, default="data", help="Path to the dataset.")
parser.add_argument("--dataset_config", type=str, default="convai2", help="Dataset configuration.")
parser.add_argument("--dataset_size", type=int, default=0, help="Number of examples to select from the dataset.")
parser.add_argument("--dataset_split", type=float, default=0.9, help="Training set size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset.")
args = parser.parse_args()

# Load tokenizer
print(f"Loading the tokenizer from {args.model_name}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Set up the tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
tokenizer.model_max_length = args.sequence_length

# Set up truncation and padding
tokenizer.set_truncation_and_padding(
    padding_strategy=PaddingStrategy.NONE,
    truncation_strategy=TruncationStrategy.LONGEST_FIRST,
    max_length=args.sequence_length,
    stride=args.stride,
    pad_to_multiple_of=8,
)

# Add assistant token to the tokenizer for the chat template because it is not in the vocabulary without a space in front of it!
tokenizer.add_tokens(["assistant"])
print(
    f"'assistant' token added to the tokenizer because it is not in the vocabulary without a space in front of it!"
)

# Add special tokens to the tokenizer
# Add "<|im_start|>", "<|im_end|>", "<|pause|>", "<|mem_start|>", "<|mem_end|>", etc.
additional_special_tokens = [
    "<|im_start|>",
    "<|im_end|>",
    "<|named_user|>",  # Named user. For future use. Example: "<|im_start|><|named_user|>Alice\n<Alice's message><|im_end|>"
    "<|named_assistant|>",  # Named assistant. For future use. Example: "<|im_start|><|named_assistant|>Assistant George\n<Assistant George's message><|im_end|>"
    "<|mem_start|>",
    "<|mem_end|>",  # Memory start and end tokens. For future use. Store hidden information in the context, e.g. "<|mem_start|>Alice's birthday is 12th May.<|mem_end|>"
    "<|pause|>",  # Pause token. For future use. See https://arxiv.org/abs/2310.02226.pdf Think before you speak: Training Language Models With Pause Tokens
]

# Add additional special tokens
if args.additional_special_tokens:
    additional_special_tokens += args.additional_special_tokens

# Add <|spare_1|>, <|spare_2|>, etc. to the tokenizer to make the vocab size a multiple of 8
for i in range(1, 8 - (len(tokenizer) + len(additional_special_tokens)) % 8 + 1):
    additional_special_tokens.append(f"<|spare_{i}|>")

if len(additional_special_tokens) > 0:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens},
        replace_additional_special_tokens=False,
    )

if args.additional_special_tokens:
    print(f"Additional special tokens added to the tokenizer.")

    # Print the token IDs of the special tokens
    for token in args.additional_special_tokens:
        print(f"{token}: {tokenizer(token)}")

# Assert that the vocab size is a multiple of 8
assert (
    len(tokenizer)
) % 8 == 0, "The vocabulary size is not a multiple of 8. Fix the padding code, dumbass!"

# Set up the chat template
if args.chat_template:
    tokenizer.chat_template = args.chat_template

# Load the dataset
print(f"Loading the dataset from {args.dataset_path} ({args.dataset_config})...")
dataset = load_dataset(args.dataset_path, args.dataset_config)


# Prepare the dataset
def prepare_dataset(
    dataset: DatasetDict, dataset_size: int, dataset_split: float, shuffle: bool = False, seed: int = 42
) -> DatasetDict:
    print("Preparing the dataset...")
    prepared_dataset = None

    # Shuffle if required
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

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
    prepared_dataset = prepared_dataset.train_test_split(
        test_size=1 - dataset_split, seed=seed, shuffle=shuffle
    )

    # Return the training and evaluation datasets
    return prepared_dataset


# Prepare the dataset
prepared_dataset = prepare_dataset(
    dataset=dataset,
    dataset_size=args.dataset_size,
    dataset_split=args.dataset_split,
    shuffle=args.shuffle,
    seed=args.seed,
)


# Function to tokenize a batch of texts
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False, padding=False, return_overflowing_tokens=True)


# Tokenize the dataset
print("Tokenizing the dataset...")
tokenized_dataset = prepared_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names,
    writer_batch_size=4096,
)

# Print the number of tokens in the dataset
print("Number of tokens in the dataset:", tokenized_dataset.num_tokens)
