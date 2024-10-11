def load_tokenize_save_dataset(dataset_name, tokenizer_name, output_path, dataset_config=None, batch_size=1000, num_proc=None, cache_dir=None):
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer

    # Load the dataset
    dataset = None
    print(f"Loading dataset {dataset_name} ({dataset_config})...")

    try:
        dataset = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    except ValueError as ve:
        if "Please use `load_from_disk` instead." in str(ve):
            dataset = load_from_disk(dataset_name)

    if dataset is None:
        raise ValueError(
            f"Could not load dataset from {dataset_name} ({dataset_config})."
        )

    # Load the tokenizer
    print(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Function to tokenize the text while keeping original columns
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(examples["text"], truncation=False)

        # Add the labels
        tokenized["labels"] = tokenized["input_ids"].copy()

        # Add tokenized data to the examples
        examples.update(tokenized)
        return examples

    # Apply the tokenization to the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=batch_size, num_proc=num_proc, desc="Tokenizing")

    # Save the tokenized dataset
    print(f"Saving tokenized dataset...")
    tokenized_dataset.save_to_disk(output_path)

    print(f"Tokenized dataset saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load, tokenize, and save a HuggingFace dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset on HuggingFace, or local path to a dataset")
    parser.add_argument("tokenizer_name", type=str, help="Name of the tokenizer on HuggingFace, or local path to a tokenizer")
    parser.add_argument("output_file", type=str, help="Path to save the tokenized dataset")

    parser.add_argument("--dataset_config", type=str, default=None, help="Configuration of the dataset on HuggingFace")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for tokenization")
    parser.add_argument("--num_proc", type=int, default=None, help="Number of processes to use for tokenization")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache the dataset")

    args = parser.parse_args()

    load_tokenize_save_dataset(
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        output_path=args.output_file,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )
