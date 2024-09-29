# transform_colums = ['text', 'label']


def format_example(batch):
    if "label" not in batch:
        return batch
    
    return {
        "text": [
            f"<|start_of_review|>{text.replace('<br />', '\n')}<|end_of_review|><|sentiment|>{'positive' if label == 1 else 'negative'}"
            for text, label in zip(batch["text"], batch["label"])
        ]
    }
