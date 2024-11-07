import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def compute_f1_score_macro(y_true, y_pred):
    """
    Calcule le F1 score macro.
    :param y_true: Labels réels
    :param y_pred: Prédictions du modèle
    :return: F1 score macro
    """
    # Récupérer les labels uniques
    labels = np.unique(y_true)

    # Initialiser les compteurs
    f1_scores = []

    for label in labels:
        # Vrais positifs
        vrai_pos = np.sum((y_true == label) & (y_pred == label))

        # Faux positifs
        faux_pos = np.sum((y_true != label) & (y_pred == label))

        # Faux négatifs
        faux_neg = np.sum((y_true == label) & (y_pred != label))

        # Accuracy et recall
        accuracy = vrai_pos / (vrai_pos + faux_pos) if (vrai_pos + faux_pos) > 0 else 0
        recall = vrai_pos / (vrai_pos + faux_neg) if (vrai_pos + faux_neg) > 0 else 0

        # F1 score
        f1 = (2 * accuracy * recall) / (accuracy + recall) if (accuracy + recall) > 0 else 0
        f1_scores.append(f1)

    # F1 score macro
    f1_macro = np.mean(f1_scores)
    return f1_macro


def remove_low_frequency_v2(inputs_documents, test_documents, labels_documents, threshold=1):
    # Compte each words
    occurrence_0 = np.sum(inputs_documents[labels_documents == 0], axis=0)
    occurrence_1 = np.sum(inputs_documents[labels_documents == 1], axis=0)

    # Différence des occurrences entre chaque classe
    difference = np.abs(occurrence_0 - occurrence_1)

    # Supprime les mots en dessous du seuil d'apparition
    delete_words = [x for x in range(len(difference)) if difference[x] < threshold]
    return np.delete(inputs_documents, delete_words, axis=1), np.delete(test_documents, delete_words, axis=1)


def remove_low_frequency_v1(inputs_documents, threshold=1):
    # Compte each words
    occurrence = np.sum(inputs_documents, axis=0)
    delete_words = [x for x in range(len(occurrence)) if occurrence[x] < threshold]
    return np.delete(inputs_documents, delete_words, axis=1)

# Define the LSTM model for text classification
class TextClassificationLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64, fc_dim=256, output_dim=1, dropout=0.5):
        super(TextClassificationLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        hidden = lstm_out[:, -1, :]  # Last LSTM output for each sequence
        fc1_out = self.fc1(hidden)
        relu_out = self.relu(fc1_out)
        dropout_out = self.dropout(relu_out)
        output = self.fc2(dropout_out)
        return self.sigmoid(output)

# Load and preprocess data
def load_and_preprocess_data(batch_size=32):
    inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
    test_documents = np.load('../../data/data_test.npy', allow_pickle=True)

    # Apply preprocessing
    threshold = 5
    inputs_documents, _ = remove_low_frequency_v2(inputs_documents, test_documents, labels_documents, threshold=threshold)

    # Convert to tensors
    X_train, X_val, y_train, y_val = train_test_split(inputs_documents, labels_documents, test_size=0.2, random_state=42)
    X_train, y_train = torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32)
    X_val, y_val = torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float32)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print("ui TRAIN")
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        print("ui VAL")
        # Validation
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_predictions = model(X_batch).squeeze()
                val_loss += criterion(val_predictions, y_batch).item()
                y_true.extend(y_batch.numpy())
                y_pred.extend((val_predictions.numpy() > 0.5).astype(int))

        print("UI RES")
        val_f1 = compute_f1_score_macro(np.array(y_true), np.array(y_pred))
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val F1 Score: {val_f1}")

# Main execution
def main():
    # Hyperparameters
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 1
    epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Load and preprocess data
    train_loader, val_loader = load_and_preprocess_data(batch_size=batch_size)
    vocab_size = next(iter(train_loader))[0].shape[1]

    # Model, criterion and optimizer
    model = TextClassificationLSTM(vocab_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("ui")
    # Training
    train_model(model, criterion, optimizer, train_loader, val_loader, epochs=epochs)

if __name__ == '__main__':
    main()