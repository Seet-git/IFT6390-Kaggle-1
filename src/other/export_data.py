import optuna
import pymysql
import pandas as pd
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

    # Sauvegarder le DataFrame en CSV
    df.to_csv('optuna_results.csv', index=False)

def export_to_csv():
    # Connexion
    connection = pymysql.connect(
        host=config.ENDPOINT,
        user=config.USER,
        password=config.PASSWORD,
        database=config.DATABASE_NAME
    )

    cursor = connection.cursor()
    cursor.execute("SHOW TABLES;")
    tables = cursor.fetchall()

    # Exporter chaque table en CSV
    for (table_name,) in tables:
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, connection)
        output_filename = f"./log/{config.OUTPUT_HP_FILENAME}.csv"
        df.to_csv(output_filename, index=False)
