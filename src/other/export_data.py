import src.config as config


def export_dict_as_python(dictionary):
    hp_file = config.OUTPUT_HP_PATH + config.OUTPUT_HP_FILENAME + ".py"
    # Open/Create file
    with open(hp_file, 'w') as file:
        for key, value in dictionary.items():
            if isinstance(value, str):
                file.write(f"{key} = '{value}'\n")
            else:
                file.write(f"{key} = {value}\n")


def export_trial_to_csv(study, trial):
    # Convert study trials to a DataFrame
    df = study.trials_dataframe()

    # Save dataframe
    df.to_csv(f'./log/optuna_{config.OUTPUT_HP_FILENAME}.csv', index=False)
