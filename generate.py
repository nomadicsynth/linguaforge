import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# import CustomLlamaModel

model_path = "./results/run-1"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path)
model = model.to(device)

# Generate text
input_text = "It is believed that the first human to walk on the moon was"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=50)

print(tokenizer.decode(output[0], skip_special_tokens=True))
