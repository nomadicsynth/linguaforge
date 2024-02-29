import torch
import transformers
from datasets import load_dataset

datasets_path = "D:/ai-stuff/datasets/"

dataset_name = "wikipedia"
subset = "20231101.en"

# Load the dataset
dataset = load_dataset(datasets_path + dataset_name, subset)

print(dataset)