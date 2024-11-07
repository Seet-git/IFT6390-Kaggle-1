import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import random

from torch.utils.data import WeightedRandomSampler


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_res(history_tab, plot_type, x_label, y_label, title, figure_name):
    # Plot settings
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=17)
    plt.ylabel(y_label, fontsize=15)
    plt.xlabel(x_label, fontsize=15)

    # Plot results
    for i in history_tab:
        plt.plot(i[plot_type])

    # Save and show the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.savefig(f"./output/{figure_name}.eps", format='eps', bbox_inches='tight')
    plt.show()
    print(f"{figure_name}.esp saved")


def plot_precision_recall_curve(model, data, plot=False):
    y_true, y_pred_proba = [], []

    # Récupération des probabilités pour chaque exemple
    model.eval()
    with torch.no_grad():
        for inputs, labels in data:
            outputs = model(inputs)
            y_pred_proba.extend(torch.sigmoid(outputs).view(-1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calcul de la courbe PRC
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall) + 0.0001

    # Plot
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label="PRC Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Meilleur seuil pour le F1-score
    best_threshold = thresholds[np.argmax(f1_scores)]

    return best_threshold

def create_balanced_sampler(y_train):
    # Count class
    class_counts = [(y_train == 0).sum().item(), y_train.sum().item()]

    weights = 1 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[y_train.long()]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler