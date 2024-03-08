import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Get command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate text from")
parser.add_argument("--model_path", type=str, required=True, help="The path to the model checkpoint")
parser.add_argument("--max_length", type=int, default=50, help="The maximum length of the generated text")
args = parser.parse_args()

model_path = args.model_path

# Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path)
model = model.to(device)

# Generate text
input_text = args.prompt
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
output = model.generate(input_ids, max_new_tokens=args.max_length)

print(tokenizer.decode(output[0], skip_special_tokens=False))
