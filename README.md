
# Classification de Texte - Bag of Words

## Table des matières

1. [Installation des dépendances](#installation-des-dépendances)
2. [Configuration de Optuna avec MySQL et Ngrok](#configuration-de-optuna-avec-mysql-et-ngrok)
3. [Utilisation de wandb](#utilisation-de-wandb)
4. [Programmes Principaux (Main) et Configuration](#programmes-principaux-main-et-configuration)
5. [Fonctionnalités de Suivi des Logs et Résultats](#fonctionnalités-de-suivi-des-logs-et-résultats)

## Installation des Dépendances

Pour commencer, assurez-vous d'installer toutes les dépendances nécessaires à ce projet. Utilisez la commande suivante :

```bash
pip install -r requirements.txt
```

Les dépendances incluent des bibliothèques telles que `numpy`, `torch`, `wandb`, `optuna`, etc.

Note: Python 3.9.19 (vérifié)

## Configuration de Optuna avec MySQL et Ngrok

### 1. Configuration de la Base de Données MySQL

Pour optimiser vos modèles avec `optuna` et stocker les résultats dans une base de données, configurez MySQL comme suit :
1. **Installer MySQL:** Commencez par installer MySQL

   - Télécharger sur Windows / Mac / Linux: https://dev.mysql.com/downloads/mysql/

   - Terminal Linux:
      ```bash
      sudo apt-get update
      sudo apt-get install mysql-server
      ```
     
   Une fois téléchargé, vous devriez pouvoir utiliser cette commande : 
   ```bash
   sudo mysql -u root -p
   ```

2. **Créer une base de données MySQL :**
   Connectez-vous à MySQL et exécutez les commandes suivantes :
   
   ```sql
   CREATE DATABASE optuna_db;
   CREATE USER 'optuna_user'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON optuna_db.* TO 'optuna_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

3. **Autorisation et Accès :**
   - Assurez-vous que `optuna_user` a les autorisations nécessaires pour créer, lire, écrire et supprimer les entrées dans `optuna_db`.
   - Configurez les accès réseau de votre base de données si nécessaire.


Si tout est correct, vous devriez voir la base de données, avec la commande suivante :
   ```sql
    SHOW GRANTS FOR 'optuna_user'@'localhost';
    SHOW databases;
   ```

### 2. Utilisation de Ngrok pour une Connexion Externe (OPTIONNEL)

Pour rendre la base de données accessible à distance (utile si vous ne pouvez pas accéder à `localhost`), vous pouvez utiliser `ngrok` :

1. **Installer Ngrok et Ouvrir une Connexion avec Ngrok :**
   ```bash
   ngrok tcp 3306
   ```
   Remplacez `3306` par le port utilisé par votre serveur MySQL si différent.

## Utilisation de wandb

Wandb est utilisé pour suivre les expériences. Avant d'exécuter un script qui utilise `wandb`, connectez-vous en utilisant la commande suivante :

```bash
wandb login
```

Si vous décidez d'activer le suivi avec `wandb` dans vos scripts, assurez-vous que l'authentification est configurée correctement.

## Programmes Principaux et Configuration

Les programmes principaux (fichiers `naives_bayes` et `neural_network`) nécessitent une configuration pour les identifiants MySQL, le modèle, d'autres options. 
Pour configurer ces fichiers :

1. **Modifier le Fichier `config.py` :**
   - Entrer vos identifiants MySQL dans la section `LOGIN MYSQL`
   - Spécifiez le modèle à utiliser, par exemple `Naives_bayes`, `MLP_v1`, `MLP_v2` ou `Perceptron`.
   - Ngrok **uniquement**: Remplacer `localhost` par l'URL générée par Ngrok   

2. **Personnalisation :**
   - Vous pouvez modifier la valeur des paramètres selon vos besoins spécifiques.

## Fonctionnalités de Suivi des Logs et Résultats

Le projet dispose de plusieurs fonctionnalités de suivi :

### Optuna

Optuna dispose d'un dashboard pour visualiser les résultats, pour y accéder, il vous suffit d'exécuter la commande suivante:
```bash
pip install optuna-dashboard
optuna-dashboard "mysql+pymysql://optuna_user:your_password@localhost:3306/optuna_db"
```

### Suivi local

1. **Enregistrement des Logs :**
   - Les logs d'entraînement et de validation sont enregistrés au fur et à mesure de l'optimisation.
   - Exemple : `src/Naives_bayes/logs/log_optimisation.csv`.

2. **Suivi des hyper-paramètres :**
   - Les meilleurs hyper-paramètres sont enregistrés dans un fichier.
   - Exemple : `src/hyperparameters/hp.py`.

3. **Correlation des hyper-paramètres :**
   - Une matrice de correlation est enregistrée dans un fichier pendant l'optimisation :
    - Exemple : `src/Neural_network/logs/matrix_corr.svg`.

4. **Visualisation des Plots :**
   - Les visualisations des courbes de perte, d'évolution des métriques et d'autres plots sont générées.
   - Exemple : `plots/ROC_curve.png`.

Ces fichiers sont créés automatiquement lors de l'exécution des scripts et peuvent être personnalisés pour inclure des métriques ou des visualisations spécifiques.