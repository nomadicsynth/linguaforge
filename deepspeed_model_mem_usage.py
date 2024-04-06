from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

model = AutoModel.from_pretrained(
    "/media/gronkomatic/Embiggen/ai-stuff/training-results/runs/run-20240330-025555/checkpoint-109698")
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1
)