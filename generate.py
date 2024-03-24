import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Get command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate text from")
parser.add_argument("--model_path", type=str, required=True, help="The path to the model checkpoint")
parser.add_argument("--max_length", type=int, default=50, help="The maximum length of the generated text")
parser.add_argument("--temperature", type=float, default=1.0, help="The temperature for sampling")
parser.add_argument("--device", type=str, default="cpu", help="The device to run the model on")
args = parser.parse_args()

model_path = args.model_path

# Set device
device = torch.device(args.device)
if device.type == "cuda" and not torch.cuda.is_available():
    raise ValueError("CUDA is not available, please run on CPU")

# Load tokenizer
# tokenizer_name = "meta-llama/Llama-2-7b-hf"
# tokenizer_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to end-of-sequence token

streamer = TextStreamer(tokenizer)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# Switch model to inference mode
model.eval()
# Move model to device
model = model.to(device)

# Generate text
input_text = args.prompt
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
_ = model.generate(input_ids, streamer=streamer, do_sample=True, max_length=args.max_length, temperature=args.temperature)
