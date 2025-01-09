# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:56:43 2025

@author: rapha
"""

# Importing the needed libraries
import pandas as pd
import os
from itertools import combinations

from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import accuracy_score, f1_score

# Function to load the data from the folder
def load_data(file_path):
    # Read the raw data with tab delimiter and specify the index column
    data = pd.read_csv(file_path, delimiter='\t', index_col=0)
    
    # Drop any leading/trailing whitespace in column names, I found some issues in some files
    data.columns = data.columns.str.strip()

    # Ensure the 'clase' column is treated as a string and use it as the label
    data['clase'] = data['clase'].astype(str).str.strip()

    # Display the number of features
    num_features = data.shape[1] - 1
    print(f"Number of features: {num_features}")

    return data

# Function to split the data into the train, validation and test sets
def split_data(df, label_column, test_size=0.3, val_size=0.5, random_state=42):
    # Remove the label
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    # Split into train and temp sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Split temp into test and validation sets
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def display_data(X, y, x_feature, y_feature):
    """
    Display a scatter plot of two features from the dataset.
    """
    plt.figure(figsize=(10, 6))
    
    for label in y.unique():
        subset = X[y == label]
        plt.scatter(subset[x_feature], subset[y_feature], label=label)
    
    plt.title(f'{x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title='Class')
    plt.grid(True)
    plt.show()

def display_all_pairs(X, y):
    """ 
    Display scatter plots for all pairs of features.
    """
    features = X.columns
    for x_feature, y_feature in combinations(features, 2):
        display_data(X, y, x_feature, y_feature)
        
# Function to open and load all the datasets
def open_files(base_dir):
    # List all folders in the base directory
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    # Some debugging
    print("Folders found:", folders)
    num = 0

    # Dictionary to hold the data for each folder
    datasets = {}
    
    # Iterate through each folder and open the specified file
    for folder in folders:
        file_name = f"{folder}_R.dat"
        file_path = os.path.join(base_dir, folder, file_name)
        if os.path.exists(file_path):
            print(f"Opening file: {file_path}")
            datasets[folder] = load_data(file_path)
            num += 1
        else:
            print(f"File not found: {file_path}")

    print(f"Number of files opened: {num}")
    return datasets

base_dir = "data"
datasets = open_files(base_dir)

# Dictionary to store accuracies and F1 scores for all datasets
results = {}

# Loop through each dataset
for folder, data in datasets.items():
    print(f"\nProcessing {folder} dataset...")
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(data, label_column='clase')
    
    # Ensure labels are numeric for kmeans
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)
    
    # Train and evaluate SVM
    classifier = SVC(kernel='linear', decision_function_shape="ovo")
    linear_ovr = classifier.fit(X_train, y_train)
    pred = linear_ovr.predict(X_test)
    f1 = f1_score(y_test, pred, average='weighted')
    accuracy = accuracy_score(y_test, pred)
    
    # Train and evaluate Random Forest
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_pred = rf_classifier.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    
    # Perform Agglomerative Clustering and map clusters to labels
    agg_cluster = AgglomerativeClustering(n_clusters=len(set(y_train)), metric='euclidean', linkage='ward')
    agg_cluster.fit(X_train)
    agg_labels = agg_cluster.fit_predict(X_test)
    agg_labels = agg_labels.astype(float)  # Ensure labels are treated as floats
    agg_accuracy = accuracy_score(y_test, agg_labels)
    agg_f1 = f1_score(y_test, agg_labels, average='weighted')
    
    # Perform K-Means Clustering
    kmeans = KMeans(n_clusters=len(set(y_train)), random_state=42, n_init=10)
    kmeans.fit(X_train)
    kmeans_labels = kmeans.predict(X_test)
    kmeans_labels = kmeans_labels.astype(float)  # Ensure labels are treated as floats
    kmeans_accuracy = accuracy_score(y_test, kmeans_labels)
    kmeans_f1 = f1_score(y_test, kmeans_labels, average='weighted')
    
    # Add results to the dictionary
    results[f"{folder} - SVM"] = {'accuracy': accuracy, 'f1_score': f1}
    results[f"{folder} - Random Forest"] = {'accuracy': rf_accuracy, 'f1_score': rf_f1}
    results[f"{folder} - Agglomerative Clustering"] = {'accuracy': agg_accuracy, 'f1_score': agg_f1}
    results[f"{folder} - K-Means Clustering"] = {'accuracy': kmeans_accuracy, 'f1_score': kmeans_f1}
    
    # Perform cross-validation for both supervised models
    cv_scores_svc = cross_val_score(classifier, X_train, y_train, cv=5)
    cv_scores_rf = cross_val_score(rf_classifier, X_train, y_train, cv=5)
    
    print('Supervised techniques:')
    print(f'{folder} SVM accuracy: {accuracy} and F1 score: {f1}')
    print(f'{folder} Random Forest accuracy: {rf_accuracy} and F1 score: {rf_f1}')
    print('Unsupervised techniques')
    print(f'{folder} Agglomerative Clustering accuracy: {agg_accuracy} and F1 score: {agg_f1}')
    print(f'{folder} K-Means Clustering accuracy: {kmeans_accuracy} and F1 score: {kmeans_f1}')
    print('Cross validation:')
    print(f'{folder} Mean SVM cross-validation score: {cv_scores_svc.mean():.2f}')
    print(f'{folder} Mean Random Forest cross-validation score: {cv_scores_rf.mean():.2f}')

# Calculate the average accuracy and F1 score for each model
def compute_average_metrics(results):
    avg_metrics = {
        'SVM': {'accuracy': [], 'f1_score': []},
        'Random Forest': {'accuracy': [], 'f1_score': []},
        'Agglomerative Clustering': {'accuracy': [], 'f1_score': []},
        'K-Means Clustering': {'accuracy': [], 'f1_score': []}
    }
    
    # Collect accuracy and F1 scores
    for key, scores in results.items():
        if 'SVM' in key:
            avg_metrics['SVM']['accuracy'].append(scores['accuracy'])
            avg_metrics['SVM']['f1_score'].append(scores['f1_score'])
        elif 'Random Forest' in key:
            avg_metrics['Random Forest']['accuracy'].append(scores['accuracy'])
            avg_metrics['Random Forest']['f1_score'].append(scores['f1_score'])
        elif 'Agglomerative Clustering' in key:
            avg_metrics['Agglomerative Clustering']['accuracy'].append(scores['accuracy'])
            avg_metrics['Agglomerative Clustering']['f1_score'].append(scores['f1_score'])
        elif 'K-Means Clustering' in key:
            avg_metrics['K-Means Clustering']['accuracy'].append(scores['accuracy'])
            avg_metrics['K-Means Clustering']['f1_score'].append(scores['f1_score'])
    
    # Calculate the mean accuracy and F1 score
    for model in avg_metrics:
        avg_metrics[model]['accuracy'] = sum(avg_metrics[model]['accuracy']) / len(avg_metrics[model]['accuracy'])
        avg_metrics[model]['f1_score'] = sum(avg_metrics[model]['f1_score']) / len(avg_metrics[model]['f1_score'])
    
    return avg_metrics

avg_metrics = compute_average_metrics(results)
print("Average Metrics for Each Classifier:")
for model, metrics in avg_metrics.items():
    print(f"{model} - Average Accuracy: {metrics['accuracy']:.4f}, Average F1 Score: {metrics['f1_score']:.4f}")