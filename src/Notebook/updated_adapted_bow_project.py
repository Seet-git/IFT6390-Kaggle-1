import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras.src.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from keras import regularizers




# Data Loading
def load_data():
    X_train = np.load('../../data/data_train.npy', allow_pickle=True)
    y_train = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
    X_test = np.load('../../data/data_test.npy', allow_pickle=True)
    vocab_map = np.load('../../data/vocab_map.npy', allow_pickle=True)
    return X_train, y_train, X_test, vocab_map


# Exploratory Data Analysis (EDA)
def perform_eda(X, y, vocab_map):
    print(f"Shape of X_train: {X.shape}")
    print(f"Shape of y_train: {y.shape}")

    # Class distribution
    class_counts = Counter(y)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

    # Word Frequency Analysis
    word_counts = np.sum(X, axis=0)
    sorted_word_indices = np.argsort(word_counts)[::-1]
    most_common_words = [(vocab_map[i], word_counts[i]) for i in sorted_word_indices[:10]]
    print("Most common words (by count):")
    for word, count in most_common_words:
        print(f"{word}: {count}")

    plt.figure(figsize=(10, 6))
    words, counts = zip(*most_common_words)
    sns.barplot(x=list(words), y=list(counts))
    plt.title('Top 10 Most Common Words')
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    plt.show()


# Model Building and Training with Regularization and Class Balancing
def build_and_train_model(X_train, y_train, X_val, y_val):
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1,
                        class_weight=class_weight_dict)
    return model, history


# Model Evaluation
def evaluate_model(model, X_val, y_val):
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation F1 Score: {f1}")
    print(f"Validation Accuracy: {acc}")

    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def main():
    # Load data
    X_train, y_train, X_test, vocab_map = load_data()

    # EDA
    perform_eda(X_train, y_train, vocab_map)

    # Train-validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Model training
    model, history = build_and_train_model(X_train_split, y_train_split, X_val, y_val)

    # Model evaluation
    evaluate_model(model, X_val, y_val)


if __name__ == '__main__':
    main()
