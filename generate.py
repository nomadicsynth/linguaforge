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
tokenizer = AutoTokenizer.from_pretrained(model_path)

streamer = TextStreamer(tokenizer)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# Switch model to inference mode
model.eval()
# Move model to device
model = model.to(device)

input_text = args.prompt

# If the tokeniser has a chat template, apply it to the input text
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    conversation_history = [{"role": "user", "content": input_text}]
    input_text = tokenizer.apply_chat_template(conversation_history, add_generation_prompt=True, tokenize=False)

# Generate text
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
_ = model.generate(input_ids, streamer=streamer, do_sample=True, max_length=args.max_length, temperature=args.temperature)
