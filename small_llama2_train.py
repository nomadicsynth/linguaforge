from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer, TrainingArguments, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset
import torch

# Model settings
hidden_layers = 8  # Number of transformer layers
hidden_size = 1024  # Size of the hidden states in the transformer layers
intermediate_size = 2048  # Size of the feed-forward network in the transformer layers
attention_heads = 32  # Number of attention heads
context_length = 1024  # Length of the input context

# Training settings
epochs = 5  # Number of training epochs
batch_size = 8  # Number of sequences to process in parallel
gradient_accumulation_steps = 10  # Number of update steps to accumulate before performing a backward pass
logging_steps = 1  # Log training loss every X steps
warmup_steps = 100 / gradient_accumulation_steps  # Number of warmup steps for the learning rate scheduler

run_dir = "./runs"
output_dir = "./results"
logging_dir = "./logs"
final_dir = "./final"

learning_rate = 5e-5
lr_scheduler_type = "linear"
optim = "adamw_torch"  # Use PyTorch's AdamW optimizer

evaluation_strategy = "epoch"
eval_steps = 0.25
save_strategy = "epoch"
save_steps = 0.25

load_best_model_at_end = True
metric_for_best_model = "loss"


# Custom model class to override the forward method
class CustomLlamaModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        # Ensure return_dict is True to work with Trainer properly
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Force return_dict to True for compatibility
            cache_position=cache_position,
        )

        # If labels are provided, calculate the loss.
        # TODO: Find out why the default loss calculation is not working, which required this custom forward method.
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Ignore the prediction for the first token since we don't have a true label for it
            loss = loss_fct(
                outputs.logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1)
            )

            return CausalLMOutputWithPast(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # If no labels, just return the original model output
        return outputs


# Ensure your GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration for a hypothetical 1B parameter model
config_1B = LlamaConfig(
    vocab_size=32000,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=hidden_layers,
    num_attention_heads=attention_heads,
    max_position_embeddings=context_length,
    pad_token_id=2,
    torch_dtype="bfloat16"
)

# Initialize the model with bfloat16 precision
model = CustomLlamaModel(config_1B)
# model = model.half()  # Convert model parameters to bfloat16
model = model.to(device)  # Move model to GPU
model = model.train()  # Set model to training mode

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token

# Prepare dataset (example using 'wikimedia/wikipedia', '20231101.en' subset)
dataset = load_dataset("D:/ai-stuff/datasets/wikipedia", "20231101.en")
small_train_dataset = dataset["train"].select(range(10000))
small_eval_dataset = dataset["train"].select(range(10000, 11000))


# Tokenize the dataset
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=context_length)

    # Shift the input ids to the left to create the labels so that the model predicts the next token.
    # The label for the last token is set to -100, so it's ignored by the loss function.
    tokenized_inputs["labels"] = [row[1:] + [-100] for row in tokenized_inputs["input_ids"]]

    return tokenized_inputs


tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval = small_eval_dataset.map(tokenize_function, batched=True)

# TrainingArguments setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    logging_dir=logging_dir,
    logging_steps=logging_steps,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=True,  # Enable mixed-precision training
    bf16_full_eval=True,  # Enable mixed-precision evaluation
    optim=optim,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Count the number of parameters in the model and print it in billions (B) or millions (M), if applicable
num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters: {num_params/1e9:.2f}B" if num_params >
      1e9 else f"Number of parameters: {num_params/1e6:.2f}M")

# Start training
trainer.train()

# Save the trained model
model.save_pretrained(output_dir)
