import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Funció per a Dense SIFT
def extract_dense_sift_features(image):
    sift = cv2.SIFT_create()
    step_size = 8
    keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(0, image.shape[0], step_size) 
                                                for x in range(0, image.shape[1], step_size)]
    _, descriptors = sift.compute(image, keypoints)
    return descriptors

# Processar imatges
def process_images(base_dir, img_size=(256, 256)):
    descriptors = []
    labels = []
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        if os.path.isdir(genre_path):
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    desc = extract_dense_sift_features(img_resized)
                    if desc is not None:
                        descriptors.append(desc)
                        labels.append(genre)
    return descriptors, labels

# Crear histogrames BoW
def create_bow(descriptors, n_clusters=500):
    all_descriptors = np.vstack(descriptors).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_descriptors)

    histograms = []
    for desc in descriptors:
        labels = kmeans.predict(desc.astype(np.float32))
        hist, _ = np.histogram(labels, bins=np.arange(n_clusters + 1))
        histograms.append(hist)

    scaler = MinMaxScaler()
    histograms = scaler.fit_transform(histograms)
    return histograms, kmeans

# Matriu de confusió
def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Matriu de Confusió")
    plt.xlabel("Etiqueta Predita")
    plt.ylabel("Etiqueta Real")
    plt.tight_layout()
    plt.show()

# Grid Search per optimitzar hiperparàmetres
def grid_search_svm(X_train, y_train):
    """
    Realitza Grid Search per trobar els millors hiperparàmetres per SVM (RBF).
    """
    print("[INFO] Configurant Grid Search per a SVM (RBF)...")
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1]
    }
    grid = GridSearchCV(
        estimator=SVC(kernel="rbf", probability=True, random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,  # Divisió per validació creuada interna
        verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"[INFO] Millors hiperparàmetres: {grid.best_params_}")
    return grid.best_estimator_

# Cross Validation per avaluar el model
def cross_validate_model(model, X, y, cv=5):
    """
    Avaluació del model amb cross-validation.
    """
    print("[INFO] Executant Cross Validation...")
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", verbose=1)
    print(f"Accuracy mitjana: {np.mean(scores):.2f}")
    print(f"Desviació estàndard: {np.std(scores):.2f}")
    return scores

# Main
if __name__ == "__main__":
    base_dir = "datasets/Data1/images_original"
    img_size = (256, 256)
    n_clusters = 500

    print("[INFO] Processant imatges...")
    descriptors, labels = process_images(base_dir, img_size)

    print("[INFO] Creant histogrames BoW...")
    X, kmeans = create_bow(descriptors, n_clusters=n_clusters)

    print("[INFO] Codificant etiquetes...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print("[INFO] Dividint dades en train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[INFO] Optimitzant hiperparàmetres amb Grid Search...")
    best_model = grid_search_svm(X_train, y_train)

    print("[INFO] Avaluant el model amb Cross Validation...")
    cross_validate_model(best_model, X, y, cv=5)

    print("[INFO] Avaluant el model optimitzat...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy final: {accuracy:.2f}")
    print("\nInforme de classificació:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Matriu de confusió
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
