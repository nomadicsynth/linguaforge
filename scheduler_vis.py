import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR, PolynomialLR
import math


def get_scheduler(scheduler_type, optimizer, num_training_steps, **kwargs):
    if scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=kwargs.get("end_factor", 0.1),
            total_iters=num_training_steps,
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=num_training_steps, eta_min=kwargs.get("eta_min", 0)
        )
    elif scheduler_type == "polynomial":
        return PolynomialLR(
            optimizer, total_iters=num_training_steps, power=kwargs.get("power", 1.0)
        )
    elif scheduler_type == "wsd":

        def lr_lambda(current_step):
            num_warmup_steps = kwargs.get("num_warmup_steps", 0)
            num_stable_steps = kwargs.get("num_stable_steps", 0)
            num_decay_steps = kwargs.get("num_decay_steps", num_training_steps)
            min_lr_ratio = kwargs.get("min_lr_ratio", 0)
            num_cycles = kwargs.get("num_cycles", 0.5)

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step < num_warmup_steps + num_stable_steps:
                return 1.0
            else:
                decay_step = current_step - num_warmup_steps - num_stable_steps
                progress = float(decay_step) / float(max(1, num_decay_steps))
                return max(
                    min_lr_ratio,
                    0.5
                    * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
                )

        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def get_lr_schedule(scheduler_type, num_training_steps, initial_lr=1e-3, **kwargs):
    model = torch.nn.Linear(1, 1)
    optimizer = AdamW(model.parameters(), lr=initial_lr)
    scheduler = get_scheduler(scheduler_type, optimizer, num_training_steps, **kwargs)

    lrs = []
    for _ in range(num_training_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    return lrs


def visualize_lr_schedulers(schedulers_config):
    plt.figure(figsize=(12, 6))

    for config in schedulers_config:
        lrs = get_lr_schedule(**config)
        plt.plot(
            lrs,
            label=f"{config['scheduler_type']} ({config['num_training_steps']} steps)",
        )

    plt.title("Learning Rate Schedule Comparison")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()


# Example usage
short_run_steps = 11000
full_run_steps = 500000
initial_lr = 2.9337e-4

schedulers_config = [
    {
        "scheduler_type": "linear",
        "num_training_steps": short_run_steps,
        "initial_lr": initial_lr,
    },
    {
        "scheduler_type": "polynomial",
        "num_training_steps": full_run_steps,
        "initial_lr": initial_lr,
        "lr_end": 1e-6,
        "power": 10.0,
    },
    {
        "scheduler_type": "cosine",
        "num_training_steps": full_run_steps,
        "initial_lr": initial_lr,
    }
]

visualize_lr_schedulers(schedulers_config)
