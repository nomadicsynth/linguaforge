import math

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from typing import Union, Dict, Optional, List, Tuple
from utils import print_if_main_process


def load_processing_script(script_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dataset_processing", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the processing script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_dataset(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    reformat_script: Optional[str] = None,
    split_sizes: Optional[Dict[str, Union[int, float]]] = None,
    split_priority: Optional[List[str]] = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 1000,
    num_proc: Optional[int] = None
) -> DatasetDict | IterableDatasetDict | Dataset | IterableDataset:
    """
    Prepare a dataset by optionally reformatting and splitting it.

    Args:
        dataset: The input dataset.
        reformat_script: Path to a Python script for reformatting the dataset.
        split_sizes: A dictionary of split names and their sizes (int or float).
        split_priority: Order in which to process splits. Defaults to keys of split_sizes.
        shuffle: Whether to shuffle the dataset before splitting.
        seed: Random seed for shuffling.
        batch_size: Batch size for processing iterable datasets.
        num_proc: Number of processes to use for processing.

    Returns:
        A DatasetDict or IterableDatasetDict containing the prepared dataset.
    """
    # Reformat the dataset if a processing script is provided
    if reformat_script:
        if isinstance(dataset, (Dataset, IterableDataset)):
            dataset = reformat_dataset(dataset, reformat_script)
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            for split_name in dataset:
                dataset[split_name] = reformat_dataset(dataset[split_name], reformat_script) # pyright: ignore[reportArgumentType]

    # Split the dataset if split sizes are provided
    if split_sizes:
        dataset = split_dataset(
            dataset,
            split_sizes,
            split_priority,
            shuffle,
            seed,
            batch_size,
            num_proc
        )

    return dataset


def split_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
    split_sizes: Dict[str, Union[int, float]],
    split_priority: Optional[List[str]] = None,
    shuffle: bool = False,
    seed: int = 42,
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
        batch_size: Batch size for processing iterable datasets.
        num_proc: Number of processes to use for processing.

    Returns:
        A DatasetDict or IterableDatasetDict containing the requested splits.
    """
    # Convert to appropriate dictionary type if necessary
    if isinstance(dataset, (Dataset, IterableDataset)):
        new_dataset = DatasetDict() if isinstance(dataset, Dataset) else IterableDatasetDict()
        new_dataset["train"] = dataset # pyright: ignore[reportArgumentType]
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
            result[split_name] = _subset_split(dataset[split_name], size, shuffle, seed)
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
        result["train"] = _subset_split(remaining, split_sizes['train'], shuffle, seed)

    return result

def _process_split_sizes(split_sizes: Dict[str, Union[int, float]], total_size: Optional[int]) -> Dict[str, int | float]:
    """Convert split sizes to integers and validate."""
    if all(isinstance(size, float) for size in split_sizes.values()):
        assert sum(split_sizes.values()) <= 1, "Float sizes must sum to <= 1"
        assert total_size is not None, "Total size must be known for fractional splitting"
        return {name: math.floor(total_size * size) for name, size in split_sizes.items()}
    elif all(isinstance(size, int) for size in split_sizes.values()):
        return split_sizes
    else:
        raise ValueError("All split sizes must be either int or float")

def _subset_split(split: Union[Dataset, IterableDataset], size: int, shuffle: bool, seed: int) -> Union[Dataset, IterableDataset]:
    """Subset an existing split based on the given strategy."""
    if isinstance(split, IterableDataset):
        return split.take(size)
    
    if size == len(split) or size == 0:
        return split
    
    if shuffle:
        return split.shuffle(seed=seed).select(range(size))
    else:
        return split.select(range(size))

def _split_dataset(dataset: Dataset, size: int) -> Tuple[Dataset, Dataset]:
    """Split a Dataset into two parts."""
    return dataset.select(range(size)), dataset.select(range(size, len(dataset)))

def _split_iterable(dataset: IterableDataset, size: int, batch_size: int, seed: int) -> Tuple[IterableDataset, IterableDataset]:
    """Split an IterableDataset into two parts."""
    return dataset.take(size), dataset.skip(size)

def reformat_dataset(dataset: Dataset | IterableDataset, reformat_script: str) -> Dataset | IterableDataset:
    processing_module = load_processing_script(reformat_script)
    if hasattr(processing_module, 'filter_fn'):
        if isinstance(dataset, Dataset):
            len_before = len(dataset)
        dataset = dataset.filter(processing_module.filter_fn, batched=True)
        if isinstance(dataset, Dataset):
            print_if_main_process(f"Filtered dataset: {len_before} -> {len(dataset)} examples")
    else:
        print_if_main_process("No 'filter_fn' found in the processing script; skipping filtering.")
    if hasattr(processing_module, 'format_example'):
        dataset = dataset.map(processing_module.format_example, batched=True)
    else:
        print_if_main_process("No 'format_example' found in the processing script; skipping formatting.")

    return dataset


