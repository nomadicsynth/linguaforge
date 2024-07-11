# LinguaForge

`LinguaForge` is a comprehensive script designed for training and fine-tuning machine learning models using  the HuggingFace [Transformers](https://huggingface.co/docs/transformers/index), [Datasets](https://huggingface.co/docs/datasets/index) and [TRL]() libraries, which offer a wide range of customization options for model training, dataset handling, and optimization.

## Features

- **Model Training**: Train models from scratch or start with pre-trained weights from any of the HuggingFace `Transformers`-based models.
- **Fine-Tuning**: Fine-tune models on specific tasks with minimal effort.
- **Dataset Handling**: Use any dataset from HuggingFace, or BYO. Flexible dataset options including subset, split, and batching.
- **Optimization**: Extensive support for different optimizers and learning rate schedulers.
- **Distributed Training**: Support for utilizing multiple CPUs and GPUs, and distributed training via `Accelerate` or `DeepSpeed`.
- **Early Stopping**: Prevent overfitting with early stopping capabilities.
- **Hyperparameter Search**: Optimize training by searching over a range of hyperparameters.

## Requirements

- Tested on Ubuntu Linux (24.04 LTS). May work on other operating systems.
- Python 3.12

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/nomadicsynth/linguaforge
    cd linguaforge
    ```

2. Create a virtual environment and install the required libraries:

    ```sh
    python -m venv .venv
    source .venv/bin/activate
    ./setup.sh
    ```

3. Login to wandb:

    ```sh
    wandb login
    ```

## Usage

### Pre-training from scratch

1. **Select Model**: Choose a model from the HuggingFace model hub to use as a template, or use a custom transformers-compatible model.
2. **Select Dataset**: Choose a dataset from the HuggingFace dataset hub or use a custom dataset.
3. **Train Model**: Train the model using the selected dataset.

   ```sh
   python linguaforge.py --project_name "MyProject" --template_model_name "bert-base-uncased" --dataset_name_or_path "my_dataset" --num_train_epochs 10
   ```

### Fine-tuning and Continued Training

1. **Select Model**: Choose a model from the HuggingFace model hub, or use a custom transformers-compatible model.
2. **Select Dataset**: Choose a dataset from the HuggingFace dataset hub or use a custom dataset.
3. **Fine-tune Model**: Fine-tune the model using the selected dataset.

   ```sh
   python linguaforge.py --project_name "MyProject" --model_name_or_path "bert-base-uncased" --dataset_name_or_path "my_dataset" --learning_rate 1e-5 --num_train_epochs 3
   ```

## Advanced Features

- **GrokFast Accelerated Grokking**: Experimental support for GrokFast, a novel training paradigm for faster grokking. See [Grokfast: Accelerated Grokking by Amplifying Slow Gradients](https://arxiv.org/abs/2405.20233) by Jaerin Lee, Bong Gyun Kang, Kihoon Kim, Kyoung Mu Lee for details of the algorithm.
- **Gradient Checkpointing**: Reduce memory usage with gradient checkpointing.
- **Optimizer Customization**: Choose from a wide range of optimizers and customize their parameters.
- **Early Stopping**: Configure patience and threshold for early stopping to prevent overfitting.

## Contributing

Contributions to `linguaforge` are welcome. Please ensure to follow the project's coding standards and submit a pull request for review.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

For more information on the Apache License 2.0, please visit [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).
