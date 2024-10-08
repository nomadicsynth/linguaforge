import pickle
import optuna

study_date = "20240409"
study_time = "174737"
studies_path = "/media/gronkomatic/Embiggen/ai-stuff/training-results/studies"
study_path = f"{studies_path}/mistral-small_hyperparameter_search-{study_date}-{study_time}"

# Load the saved study
with open(f"{study_path}/optuna_study.pkl", "rb") as f:
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

# Visualize the study
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_slice(study).show()
optuna.visualization.plot_parallel_coordinate(study).show()
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_contour(study).show()
optuna.visualization.plot_edf(study).show()
optuna.visualization.plot_intermediate_values(study).show()
