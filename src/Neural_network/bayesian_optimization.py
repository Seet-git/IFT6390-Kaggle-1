import urllib.parse

import optuna
import wandb
from datetime import datetime
import pytz

import src.config as config
from src.Neural_network.training import evaluation
from src.other.export_data import export_dict_as_python, export_trial_to_csv

montreal_timezone = pytz.timezone('America/Montreal')
current_time = datetime.now(montreal_timezone).strftime("%m/%d-%H:%M:%S")

global_best_score = -float('inf')


def objective(trial):
    """
    Function to capture
    :param trial:
    :return:
    """
    global global_best_score

    hyperparameters_dict = {
        "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
        "hidden_layer1": trial.suggest_int("hidden_layer1", 512, 2048, step=64),
        "hidden_layer2": trial.suggest_int("hidden_layer2", 128, 320, step=64),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 20, 60),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "minority_weight": trial.suggest_float("minority_weight", 1.0, 4.0),
        "low_frequency": trial.suggest_int("low_frequency", 0, 10),
        "high_frequency": trial.suggest_int("high_frequency", 0, 10),
        "infer_threshold": trial.suggest_float("infer_threshold", 0.1, 0.5)
    }

    # Initialisation wandb
    if config.WANDB_ACTIVATE:
        wandb.init(project="MLP Optimizer",
                   name=f"{current_time} - Trial_{trial.number}",
                   config=hyperparameters_dict
                   )

    # Evaluation
    mean_f1 = evaluation(hyperparameters_dict)

    # Update best score
    if mean_f1 > global_best_score:
        global_best_score = mean_f1
        export_dict_as_python(hyperparameters_dict)
        print(f"New best F1-score: {mean_f1}")

    # Return mean score
    return mean_f1


def bayesian_optimization(n_trials: int) -> None:
    """
    :param n_trials:
    :return:
    """
    storage_url = f"mysql+pymysql://{config.USER}:{urllib.parse.quote(config.PASSWORD)}@{config.ENDPOINT}/{config.DATABASE_NAME}"
    print(storage_url)
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=f"MLP Optimizer - {current_time}",
        sampler=optuna.samplers.TPESampler(seed=1)
    )

    study.optimize(objective, n_trials=n_trials, callbacks=[export_trial_to_csv])

    # Show results
    print("Best trial:")
    trial = study.best_trial
    print(f"\tValue: {trial.value}")
    print("\tParams:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    print("Best F1 score: ", study.best_value)
