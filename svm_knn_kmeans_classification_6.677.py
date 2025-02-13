import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Load dataset
def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset not found: {filename}. Please make sure the dataset is in the correct directory.")
    df = pd.read_csv(filename)
    return df

# Preprocess dataset
def preprocess_data(df):
    df = df[df["Class"] <= 2]  # Filtering classes 1 and 2
    X = df[["Area", "Perimeter", "Compactness", "Length of kernel", "Width of Kernel", "Asymmetry Coeffcient", "Length of Kernel Groove"]].values
    y = df["Class"].values
    return X, y

# Train and evaluate SVM models
def train_svm(X_train, y_train, X_test, y_test, kernel, degree=3):
    model = svm.SVC(kernel=kernel, degree=degree)
    y_pred = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"SVM ({kernel}) Accuracy: {accuracy:.2%}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"SVM ({kernel}) Confusion Matrix")
    plt.show()
    return accuracy

# Train and evaluate KNN
def train_knn(X_train, y_train, X_test, y_test, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    y_pred = knn.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"KNN (k={k}) Accuracy: {accuracy:.2%}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"KNN (k={k}) Confusion Matrix")
    plt.show()
    return accuracy

# Train and evaluate K-Means clustering
def train_kmeans(X, y, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=123, init='random')
    y_clusters = kmeans.fit_predict(X)
    accuracy = accuracy_score(y, y_clusters)
    
    print(f"K-Means Clustering Accuracy: {accuracy:.2%}")
    return accuracy

# Main script execution
if __name__ == "__main__":
    file_path = 'seeds_dataset_1.csv'  # Ensure this file is present
    df = load_data(file_path)
    X, y = preprocess_data(df)
    
    # Splitting data for classification models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    
    # Scaling data for distance-based models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Evaluate models
    results = {}
    results["Linear SVM"] = train_svm(X_train, y_train, X_test, y_test, kernel='linear')
    results["Gaussian SVM"] = train_svm(X_train, y_train, X_test, y_test, kernel='rbf')
    results["Polynomial SVM"] = train_svm(X_train, y_train, X_test, y_test, kernel='poly', degree=3)
    results["KNN"] = train_knn(X_train, y_train, X_test, y_test, k=5)
    results["K-Means Clustering"] = train_kmeans(X, y)
    
    print("Final Model Accuracy Scores:", results)
