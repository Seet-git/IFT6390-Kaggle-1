import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(y_data):
    count_class = y_data.value_counts()
    plt.bar(count_class.index, count_class.values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(count_class.index, ['Class 0', 'Class 1'])
    plt.show()


def undersample_data(x_inputs, y_labels):
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    x, y = rus.fit_resample(x_inputs, y_labels)

    # Save_data
    y.index.name = 'ID'
    np.save('../../data/undersampled_data.npy', x)
    y.to_csv('../data/undersampled_labels.csv')


def smote_balanced_data(x_inputs, y_labels):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy='minority')
    x, y = smote.fit_resample(x_inputs, y_labels)
    y.index.name = 'ID'
    np.save('../../data/smote_data.npy', x)
    y.to_csv('../data/smote_labels.csv')


def hybrid_data(x_inputs, y_labels):
    from imblearn.combine import SMOTEENN

    smote_enn = SMOTEENN()
    x, y = smote_enn.fit_resample(x_inputs, y_labels)
    y.index.name = 'ID'
    np.save('../../data/hybrid_data.npy', x)
    y.to_csv('../data/hybrid_labels.csv')


def main():
    # Load data
    y = pd.read_csv('../../data/label_train.csv')["label"]
    x = np.load('../../data/data_train.npy', allow_pickle=True)

    # Plot distribution
    plot_class_distribution(y)

    # Balanced data
    smote_balanced_data(x, y)
    undersample_data(x, y)
    hybrid_data(x, y)

    # Load oversampling data
    y_balanced = pd.read_csv('../../data/smote_labels.csv')["label"]
    plot_class_distribution(y_balanced)

    # Load undersampling data
    y_balanced = pd.read_csv('../../data/undersampled_labels.csv')["label"]
    plot_class_distribution(y_balanced)

    # Load hybrid data
    y_balanced = pd.read_csv('../../data/hybrid_labels.csv')["label"]
    plot_class_distribution(y_balanced)


if __name__ == "__main__":
    main()
