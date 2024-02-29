from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer, TrainingArguments, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset
import torch


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

        # If labels are provided, calculate the loss
        # if labels is not None:
        #     # Shift the logits and labels for loss calculation
        #     shift_logits = outputs.logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Calculate loss
        #     loss_fct = torch.nn.CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if labels is not None:
            # Directly use outputs for loss calculation without shifting
            loss_fct = torch.nn.CrossEntropyLoss()
            # Ignore the prediction for the first token since we don't have a true label for it
            loss = loss_fct(outputs.logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size), labels[:, 1:].contiguous().view(-1))

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
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    max_position_embeddings=2048,
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
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)

    # Shift the input ids to the left to create the labels so that the model predicts the next token.
    # The label for the last token is set to -100, so it's ignored by the loss function.
    tokenized_inputs["labels"] = [row[1:] + [-100] for row in tokenized_inputs["input_ids"]]

    return tokenized_inputs


tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval = small_eval_dataset.map(tokenize_function, batched=True)

# TrainingArguments setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=100,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    logging_dir="./logs",
    logging_steps=1,
    evaluation_strategy="epoch",
    eval_steps=0.25,
    save_strategy="epoch",
    save_steps=0.25,
    # load_best_model_at_end=True,
    # metric_for_best_model="loss",
    gradient_accumulation_steps=1,
    bf16=True,  # Enable mixed-precision training
    bf16_full_eval=True,  # Enable mixed-precision evaluation
    optim="adamw_torch",  # Use PyTorch's AdamW optimizer
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# Start training
trainer.train()

# Save the trained model
model.save_pretrained("./custom_llama_1B_model")
