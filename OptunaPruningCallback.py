import math
import optuna
from transformers import TrainerCallback

# Custom callback for Optuna pruning
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial, monitor: str):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Retrieve the metric to monitor
        eval_loss = metrics.get('eval_loss')

        # Check if eval_loss is NaN or INF
        if eval_loss is None:
            raise ValueError("The monitored metric 'eval_loss' was not found.")
        if math.isnan(eval_loss) or math.isinf(eval_loss):
            message = f"Trial was pruned at epoch {state.epoch} due to NaN or INF in eval_loss."
            raise optuna.exceptions.TrialPruned(message)

        # Report the current metric value to Optuna and check for pruning
        self.trial.report(eval_loss, step=state.epoch)
        if self.trial.should_prune():
            message = f"Trial was pruned at epoch {state.epoch}."
            raise optuna.exceptions.TrialPruned(message)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Retrieve the metrics to monitor
        grad_norm = logs.get('grad_norm')
        loss = logs.get('loss')

        # Check if we are in eval and grad_norm or loss is not available
        if grad_norm is None or loss is None:
            return

        # Check if grad_norm or loss is NaN or INF
        if math.isnan(grad_norm) or math.isinf(grad_norm) or math.isnan(loss) or math.isinf(loss):
            message = f"Trial was pruned at step {state.global_step} due to NaN or INF in grad_norm or loss."
            raise optuna.exceptions.TrialPruned(message)

        # Report the current metric value to Optuna and check for pruning
        self.trial.report(loss, step=state.global_step)
        if self.trial.should_prune():
            message = f"Trial was pruned at step {state.global_step}."
            raise optuna.exceptions.TrialPruned(message)
