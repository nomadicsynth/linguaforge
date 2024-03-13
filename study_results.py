import pickle

# Load the saved study
with open(f"./results/mistral-small_hyperparameter_search-attention_heads-8-32-1000/optuna_study.pkl", "rb") as f:
    study = pickle.load(f)

# Analyze the study
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Best trial:")
trial = study.best_trial
print("    Value: ", trial.value)
print("    Params: ")
for key, value in trial.params.items():
    print(f"      {key}: {value}")

print(study.trials_dataframe())