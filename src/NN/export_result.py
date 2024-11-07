import pymysql
import pandas as pd

# Connexion à la base de données MySQL
connection = pymysql.connect(
    host='localhost',
    user='optuna_seet',
    password='@g3NYkke*eAFRs',
    database='optuna_MLP'  # Remplacez par votre base de données
)

# Créer un curseur pour obtenir la liste des tables
cursor = connection.cursor()
cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()

# Exporter chaque table en CSV
for (table_name,) in tables:  # Chaque élément est un tuple, donc on utilise la syntaxe (table_name,)
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, connection)
    df.to_csv(f'{table_name}.csv', index=False)
    print(f"Table {table_name} exportée dans {table_name}.csv")

# Fermeture du curseur et de la connexion
cursor.close()
connection.close()
