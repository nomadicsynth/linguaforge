import optuna
import matplotlib.pyplot as plt


def objective_float(trial):
    return trial.suggest_float("learning_rate", 1e-5, 1e-1)


def objective_loguniform(trial):
    return trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)


# Create studies
study_float = optuna.create_study()
study_loguniform = optuna.create_study()

# Optimize (sample) 1000 times for each
study_float.optimize(objective_float, n_trials=1000)
study_loguniform.optimize(objective_loguniform, n_trials=1000)

# Collect results
float_values = [trial.params["learning_rate"] for trial in study_float.trials]
loguniform_values = [trial.params["learning_rate"] for trial in study_loguniform.trials]

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.hist(float_values, bins=50)
plt.title("suggest_float()")
plt.xlabel("Learning Rate")
plt.ylabel("Frequency")
plt.xscale("log")

plt.subplot(122)
plt.hist(loguniform_values, bins=50)
plt.title("suggest_loguniform()")
plt.xlabel("Learning Rate")
plt.ylabel("Frequency")
plt.xscale("log")

plt.tight_layout()
plt.show()
