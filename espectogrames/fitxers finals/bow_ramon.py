import os
import cv2
import numpy as np
import json
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Dense SIFT
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

# Entrenament i avaluació
def train_and_evaluate(models, X_train, X_test, y_train, y_test, results):
    for model, name in models:
        print(f"[INFO] Entrenant el model: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcular mètriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Desa els resultats
        results[name] = {
            "accuracy": round(accuracy, 5),
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "f1_score": round(f1, 5),
        }

        print(f"[RESULTS] {name} -> Accuracy: {accuracy:.2f}, F1-score: {f1:.2f}")

# Guardar resultats en un JSON
def save_results_to_json(results, filename="espectogrames/bow_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Resultats guardats a {filename}")

# Main
if __name__ == "__main__":
    base_dir = "datasets/Data1/images_original"
    img_size = (256, 256)
    n_clusters = 500
    results = {}

    print("[INFO] Processant imatges...")
    descriptors, labels = process_images(base_dir, img_size)

    print("[INFO] Creant histogrames BoW...")
    X, kmeans = create_bow(descriptors, n_clusters=n_clusters)

    print("[INFO] Codificant etiquetes...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print("[INFO] Dividint dades en train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[INFO] Entrenant i avaluant models...")
    models = [
        (SVC(kernel="rbf", probability=True, random_state=42), "SVM (RBF)"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        (GradientBoostingClassifier(n_estimators=100, random_state=42), "Gradient Boosting"),
        (GaussianNB(), "Gaussian Naive Bayes"),
        (KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree"),
    ]
    train_and_evaluate(models, X_train, X_test, y_train, y_test, results)

    print("[INFO] Guardant resultats...")
    save_results_to_json(results)
    print("[SUCCESS] Processament complet!")
