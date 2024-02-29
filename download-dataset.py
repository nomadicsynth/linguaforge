from datasets import load_dataset

datasets_path = "D:/ai-stuff/datasets/"

dataset_name = "wikimedia/wikipedia"
subset = "20231101.en"

# Download the dataset and save it to a new directory in the datasets folder
load_dataset(dataset_name, subset, data_dir=datasets_path)
