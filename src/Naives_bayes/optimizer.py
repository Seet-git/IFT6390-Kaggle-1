from types import SimpleNamespace
import optuna
import src.config as config
from src.Naives_bayes.training_and_saving import train_model
from src.other.export_data import export_dict_as_python, export_trial_to_csv
from datetime import datetime
import pytz

montreal_timezone = pytz.timezone('America/Montreal')
current_time = datetime.now(montreal_timezone).strftime("%m/%d-%H:%M:%S")

global_best_score = -float('inf')


def objective(trial):
    global global_best_score

    # Hyper-paramètres
    hyperparameters_dict = {
        "smoothing": trial.suggest_float('smoothing', 0.0, 1.0),
        "low_threshold": trial.suggest_int('low_threshold', 1, 20),
        "high_threshold": trial.suggest_int('high_threshold', 1, 20),
    }

    hp = SimpleNamespace(**hyperparameters_dict)

    # Entraînement et évaluation du modèle
    mean_f1_score = train_model(hp)[2]

    if mean_f1_score > global_best_score:
        global_best_score = mean_f1_score
        export_dict_as_python(hyperparameters_dict)
        print(f"New best F1-score: {mean_f1_score}")

    # Retourner le score pour l'optimisation
    return mean_f1_score


def optimize(n_trials):
    print(f"device: {config.DEVICE}")

    # Lancer l'optimisation bayésienne
    study = optuna.create_study(direction='maximize',
                                study_name=f"{config.ALGORITHM} Optimizer - {current_time}"
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
