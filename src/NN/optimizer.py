import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import optuna

from src.NN.models import *
from src.NN.utils import *
from src.Naives_bayes.preprocessing import remove_low_frequency_v2

from datetime import datetime
import pytz

# Set time zone
montreal_timezone = pytz.timezone('America/Montreal')
current_time = datetime.now(montreal_timezone).strftime("%m/%d-%H:%M:%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def save_model_predictions(model, data_test, threshold=0.5):
    # Demande du nom du fichier
    name = input("File name (Press enter > default): ")
    if name == '':
        name = "mlp_optimizer"

    # Évaluation du modèle en mode prédiction
    model.eval()
    predictions = []

    # DataLoader pour les données de test
    test_tensor = torch.tensor(data_test, dtype=torch.float32)
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False)

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs).view(-1) > threshold).float()
            predictions.extend(preds.cpu().numpy())

    # Sauvegarde des prédictions
    df_pred = pd.DataFrame(predictions, columns=['label'])
    df_pred['label'] = df_pred['label'].astype(int)
    df_pred.index.name = 'ID'
    df_pred.to_csv(f'../../output/{name}.csv')


def compute_loss(model, data):
    total_loss = 0.0
    # Mini-Batch
    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)
        # Model prediction
        pred = model.forward(inputs)

        # Compute loss
        loss = model.criterion(pred.view(-1), labels.view(-1))
        total_loss += loss.item()

    return total_loss / len(data)


def infer(model, data, threshold: float):
    # Évaluation du modèle
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in data:
            # Compute the prediction
            outputs = model(inputs)

            # Compute prediction
            pred = (torch.sigmoid(outputs).view(-1) > threshold).float()

            # Add prediction
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calcul du F1-score
    return f1_score(y_true, y_pred, average="macro")


def fit(model, train_loader, test_loader, epochs, threshold):
    model.to(device)
    # Loop on all epochs
    for epoch in range(epochs):
        model.train()

        # Mini-Batch
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Reset backward
            model.optimizer.zero_grad()

            # Model prediction
            pred = model.forward(inputs)

            # Compute loss
            loss = model.criterion(pred.view(-1), labels.view(-1))

            # Backpropagation
            loss.backward()

            # Change weights
            model.optimizer.step()

        # Compute loss
        train_loss = compute_loss(model, train_loader)
        test_loss = compute_loss(model, test_loader)

        # F1 score
        f1_score_train = infer(model, train_loader, threshold)
        f1_score_test = infer(model, test_loader, threshold)


def split_dataset(inputs_train: np.array, labels_train: np.array):
    """
    Split the dataset into training and test set
    :param inputs_train:
    :param labels_train:
    :return:
    """
    # Shuffle the dataset
    indices = np.arange(len(inputs_train))
    np.random.shuffle(indices)

    inputs_shuffled = inputs_train[indices]
    labels_shuffled = labels_train[indices]

    # Get the size of the training set
    # train_size = int(np.ceil(0.8 * len(inputs_train)))
    train_size = len(inputs_train) - 2355

    # Train set
    set_train_inputs = inputs_shuffled[0:train_size, :]
    set_train_labels = labels_shuffled[0:train_size]

    # Test set
    set_test_inputs = inputs_shuffled[train_size:, :]
    set_test_labels = labels_shuffled[train_size:]

    return set_train_inputs, set_train_labels, set_test_inputs, set_test_labels


def evaluation(inputs_documents: torch.Tensor, labels_documents: torch.Tensor, batch_size=128, hidden_layer=256,
               learning_rate=0.0001, weight_decay=0.01, epochs=10, threshold=0.5,
               minority_weight=4.0, optimizer='Adam', balanced=True):
    # Define model
    model = MLPClassifier(inputs_documents.shape[1], hidden_layer).to(device)

    # Loss function
    class_0_count = (labels_documents == 0).sum().item()
    class_1_count = labels_documents.sum().item()
    value = torch.tensor([(class_0_count / class_1_count) - minority_weight], dtype=torch.float32).to(device)
    weight = torch.maximum(torch.tensor(1), value)
    model.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    # Set optimizer
    optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        'Adagrad': optim.Adagrad,
        'Adadelta': optim.Adadelta,
        'Nadam': optim.NAdam
    }
    model.optimizer = optimizers[optimizer](model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Split dataset to train and test
    inputs_train, labels_train, inputs_test, labels_test = split_dataset(inputs_documents, labels_documents)

    # Train dataset
    train_dataset = TensorDataset(inputs_train, labels_train)

    # Balanced dataset
    if balanced:
        sampler = create_balanced_sampler(labels_train.to(device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test dataset
    test_dataset = TensorDataset(inputs_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Entraînement
    fit(model, train_loader, test_loader, epochs, threshold)

    # F1 score
    f1_score_test = infer(model, test_loader, threshold)
    print(f1_score_test)
    return model, f1_score_test


def bayesian_optimization(trial):
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
    hidden_layer = trial.suggest_int("hidden_layer", 64, 512, step=64)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 5, 100)
    threshold = trial.suggest_float("threshold", 0.1, 0.9)
    minority_weight = trial.suggest_float("minority_weight", 1.0, 4.0)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "Adagrad", "RMSprop", "Adadelta", "Nadam"])
    balanced = trial.suggest_categorical("balanced", [True, False])

    score = evaluation(
        x_tensor.to(device), y_tensor.to(device),
        batch_size=batch_size,
        hidden_layer=hidden_layer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        threshold=threshold,
        minority_weight=minority_weight,
        optimizer=optimizer,
        balanced=balanced
    )
    return score


def launch_evaluation():
    # storage_url = "mysql+pymysql://optuna_seet:%40g3NYkke%2AeAFRs@localhost/optuna_MLP" # Local
    storage_url = "mysql+pymysql://optuna_seet:%40g3NYkke%2AeAFRs@2.tcp.ngrok.io:10594/optuna_MLP"  # Google colab - server Mysql - grock

    # Bayesian optimization
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=f"MLP Optimizer - {current_time}"
    )
    study.optimize(bayesian_optimization, n_trials=500)

    print("Best trial:")
    trial = study.best_trial
    print(f"\tValue: {trial.value}")
    print("\tParams:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
    test_documents = np.load('../../data/data_test.npy', allow_pickle=True)

    inputs_documents, test_documents = remove_low_frequency_v2(inputs_documents, test_documents, labels_documents,
                                                               threshold=3)
    x_tensor = torch.tensor(inputs_documents, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(labels_documents, dtype=torch.float32).to(device)

    # launch_evaluation()
    # _, score = evaluation(x_tensor, y_tensor, batch_size=160, hidden_layer=384,
    #                           learning_rate=0.0484551386562176,
    #                           weight_decay=0.0005996759054919, epochs=24, threshold=0.2697006774836787,
    #                           minority_weight=2.191867216366139,
    #                           optimizer="Adadelta", balanced=False)
    # mean_score_1 += score
    set_seed(1)
    model, score = evaluation(x_tensor, y_tensor, batch_size=192, hidden_layer=320, learning_rate=0.0003633669014150335,
                          weight_decay=.0023993097912930173, epochs=46, threshold=0.5657014255006196,
                          minority_weight=2.809875574543455,
                          optimizer="Adagrad", balanced=True)


    save_model_predictions(model, test_documents, 0.5657014255006196)
