import os
import time
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import threading

# Funció per crear el diccionari visual
class VisualBoW:
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.dictionary = None

    def fit(self, descriptors):
        """Entrena el model KMeans per crear el diccionari visual."""
        all_descriptors = np.vstack(descriptors)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)
        self.dictionary = self.kmeans.cluster_centers_

    def transform(self, descriptors):
        """Crea un histograma BoW per a cada descriptor."""
        histograms = []
        for desc in descriptors:
            labels = self.kmeans.predict(desc)
            hist, _ = np.histogram(labels, bins=np.arange(self.n_clusters + 1))
            histograms.append(hist)
        return np.array(histograms)

def extract_sift_features(image):
    """Extreu descriptors SIFT per a una imatge."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def processament_bow(base_dir, img_size=(128, 128), n_clusters=100):
    """Processa les dades utilitzant la tècnica Bag of Words."""
    data = []
    labels = []
    descriptors = []
    discarded = 0

    # Recórrer les subcarpetes de generes musicals
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        if os.path.isdir(genre_path):  # Comprovar que és una carpeta
            print(f"[INFO] Processant gènere: {genre}")
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    desc = extract_sift_features(img_resized)
                    if desc is not None:
                        descriptors.append(desc)
                        data.append(img_resized)
                        labels.append(genre)
                    else:
                        discarded += 1
                else:
                    print(f"[WARNING] Fitxer no trobat o invàlid: {img_path}")

    print(f"[INFO] Imatges descartades per descriptors buits: {discarded}")

    # Crear el diccionari visual
    bow = VisualBoW(n_clusters=n_clusters)
    bow.fit(descriptors)
    histograms = bow.transform(descriptors)

    # Normalitzar els histogrames
    scaler = MinMaxScaler()
    histograms = scaler.fit_transform(histograms)

    return histograms, labels


def codificar_label(labels):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return y, label_encoder

def model_assess_to_json_timer(model, X_train, X_test, y_train, y_test, title, resultats, timeout=360):
    """
    Entrena i avalua un model amb control de temps. Si el model supera el temps límit (timeout),
    l'entrenament s'interromp.
    """
    if title not in resultats:
        resultats[title] = {}

    def entrenar_model():
        """
        Funció per ajustar el model en un thread separat.
        """
        print(f"[INFO] Entrenant el model {title}...")
        try:
            model.fit(X_train, y_train)
            print(f"[SUCCESS] Model {title} entrenat correctament!")
        except Exception as e:
            print(f"[ERROR] Entrenament del model {title} fallat: {str(e)}")
            resultats[title]["Error"] = f"Error durant l'entrenament: {str(e)}"
            raise

    # Crear un thread per entrenar el model
    print(f"[INFO] Iniciant el thread d'entrenament per al model {title}...")
    training_thread = threading.Thread(target=entrenar_model)
    training_thread.start()

    # Esperar que el thread acabi dins del temps límit
    training_thread.join(timeout)
    if training_thread.is_alive():
        print(f"[TIMEOUT] Model {title} interromput després de {timeout / 60:.1f} minuts.")
        resultats[title]["Error"] = f"Entrenament interromput després de {timeout / 60:.1f} minuts."
        return  # Interrompre si s'excedeix el temps

    # Si l'entrenament acaba, continuar amb l'avaluació
    print(f"[INFO] Avaluant el model {title}...")
    try:
        start_predict = time.time()
        preds = model.predict(X_test)
        predict_time = round(time.time() - start_predict, 5)
        print(f"[SUCCESS] Prediccions fetes pel model {title}!")

        # Calcular mètriques
        accuracy_test = round(accuracy_score(y_test, preds), 5)
        precision_test = round(precision_score(y_test, preds, average="weighted", zero_division=0), 5)
        recall_test = round(recall_score(y_test, preds, average="weighted", zero_division=0), 5)
        f1_test = round(f1_score(y_test, preds, average="weighted", zero_division=0), 5)

        if len(np.unique(y_test)) > 1 and hasattr(model, "predict_proba"):
            roc_auc_test = round(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"), 5)
        else:
            roc_auc_test = None

        # Desa els resultats
        resultats[title]["accuracy"] = accuracy_test
        resultats[title]["precision"] = precision_test
        resultats[title]["recall"] = recall_test
        resultats[title]["f1_score"] = f1_test
        if roc_auc_test is not None:
            resultats[title]["roc_auc"] = roc_auc_test
        resultats[title]["temps_predict"] = predict_time
        print(f"[SUCCESS] Mètriques calculades per al model {title}!")

    except Exception as e:
        print(f"[ERROR] Avaluació del model {title} fallada: {str(e)}")
        resultats[title]["Error"] = f"Error durant l'avaluació: {str(e)}"

def guardar_resultats_a_json(resultats, nom_fitxer="bow_results.json"):
    """Guarda els resultats en un fitxer JSON."""
    with open(nom_fitxer, "w") as fitxer:
        json.dump(resultats, fitxer, indent=4)
    print(f"Resultats guardats a {nom_fitxer}")

if __name__ == "__main__":
    base_dir = "ACproject-14-grup/datasets/Data1/images_original"
    resultats = {}

    print("[INFO] Processant dades amb Bag of Words...")
    X, labels = processament_bow(base_dir, n_clusters=100)
    y, label_encoder = codificar_label(labels)

    print("[INFO] Dividint dataset en conjunt train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)

    print("[INFO] Entrenant els models...")
    models = [
        
    ]
    models = [
        (LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression"),
        (SVC(kernel="rbf", probability=True, random_state=42), "Support Vector Machine (SVM)"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        (GaussianNB(), "Gaussian Naive Bayes"),
        (BernoulliNB(), "Bernoulli Naive Bayes"),
        (MultinomialNB(), "Multinomial Naive Bayes"),
        (KNeighborsClassifier(n_neighbors=19), "K-Nearest Neighbors"),
        (DecisionTreeClassifier(), "Decision Trees"),
        (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0), "Gradient Boosting"),
        (XGBClassifier(n_estimators=1000, learning_rate=0.05), "Cross Gradient Booster"),
        (XGBRFClassifier(objective= 'multi:softmax'),"Cross Gradient Booster (Random Forest)" )

    ]

    for model, title in models:
        print(f"[INFO] Començant l'entrenament per al model {title}...")
        model_assess_to_json_timer(model, X_train, X_test, y_train, y_test, title, resultats)

    print("[INFO] Guardant els resultats...")
    guardar_resultats_a_json(resultats)
    print("[SUCCESS] Resultats guardats correctament!")
