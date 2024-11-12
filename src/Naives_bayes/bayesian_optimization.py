import urllib.parse
from types import SimpleNamespace
import optuna
import config
from src.Naives_bayes.predict import train_model
from src.scripts.export_data import export_dict_as_python, export_trial_to_csv
from datetime import datetime
import pytz

from src.scripts.matrix_hyperparameters import plot_hyperparameter_correlation_matrix

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
    storage_url = f"mysql+pymysql://{config.USER}:{urllib.parse.quote(config.PASSWORD)}@{config.ENDPOINT}/{config.DATABASE_NAME}"
    # Lancer l'optimisation bayésienne
    study = optuna.create_study(direction='maximize',
                                storage=storage_url,
                                study_name=f"{config.ALGORITHM} Optimizer - {current_time}"
                                )
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[export_trial_to_csv, plot_hyperparameter_correlation_matrix])

    # Show results
    print("Best trial:")
    trial = study.best_trial
    print(f"\tValue: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"\t{key}: {value}")

    print("Best F1 score: ", study.best_value)
