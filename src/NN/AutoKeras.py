import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import autokeras as ak
from src.NN.optimizer import f1_score


# Charger les données
train_data = np.load('../../data/data_train.npy', allow_pickle=True)
train_labels =  pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
test_data = np.load('../../data/data_test.npy', allow_pickle=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.25)

input_node = ak.Input()
output_node = ak.ClassificationHead(num_classes=2, metrics=[f1_score])

auto_model = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    max_trials=5  # Nombre d'essais pour trouver le meilleur modèle
)

auto_model.fit(x_train, y_train, epochs=10)

# Prédictions
predicted_labels = auto_model.predict(test_data)

print(f1_score(y_test, predicted_labels))

# Enregistrement du meilleur modèle
model = auto_model.export_model()
model.save('best_autokeras_model')
