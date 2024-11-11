import numpy as np
from scipy.stats import ks_2samp

# Charger les données
data_train = np.load('../../data/undersampled_data.npy', allow_pickle=True)
data_test = np.load('../../data/data_test.npy', allow_pickle=True)

# Aplatir les données
data_train_flat = data_train.flatten()
data_test_flat = data_test.flatten()

# Définir une taille d'échantillon
sample_size = 2300  # Exemple de taille d'échantillon

# Échantillonner aléatoirement si les données sont trop volumineuses
if len(data_train_flat) > sample_size:
    data_train_flat = np.random.choice(data_train_flat, size=sample_size, replace=False)

if len(data_test_flat) > sample_size:
    data_test_flat = np.random.choice(data_test_flat, size=sample_size, replace=False)

# Effectuer le test de Kolmogorov-Smirnov
ks_statistic, p_value = ks_2samp(data_train_flat, data_test_flat)

# Afficher les résultats
print(f"Statistic de Kolmogorov-Smirnov: {ks_statistic}")
print(f"P-value: {p_value}")

# Interpréter le résultat
alpha = 0.05  # niveau de signification
if p_value < alpha:
    print("Nous rejetons l'hypothèse nulle : les distributions sont différentes.")
else:
    print("Nous ne pouvons pas rejeter l'hypothèse nulle : les distributions sont similaires.")


