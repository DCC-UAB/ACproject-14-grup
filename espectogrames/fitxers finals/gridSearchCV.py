import os
import time
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def augment_image(image):
    flipped = np.fliplr(image)
    noisy = np.clip(image + np.random.normal(0, 0.02, image.shape), 0, 1)  
    brighter = np.clip(image * 1.2, 0, 1)
    darker = np.clip(image * 0.8, 0, 1)
    return [image, flipped, noisy, brighter, darker]

def processament(data, labels, base_dir, img_size=(128, 128)):
    """
    Processa imatges des d'un directori i extreu característiques bàsiques.
    """
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        if os.path.isdir(genre_path):
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size) / 255.0
                    data.append(img_resized.flatten())
                    labels.append(genre)


def codificar_label(data):
    """
    Codifica les etiquetes amb LabelEncoder.
    """
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data, label_encoder


def definirXY_normalitzar(data):
    """
    Divideix el dataset en X i y i aplica normalització.
    """
    X = data.drop(['label'], axis=1)
    y = data['label']
    columnes = X.columns
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=columnes)
    return X, y


def cross_validation(model, X, y):
    """
    Aplica cross-validation a un model.
    """
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"[INFO] Accuracy mitjà amb cross-validation: {np.mean(cv_scores):.4f}")
    return cv_scores


def grid_search_with_cv_and_final_validation(X, y, resultats, models):
    """
    Aplica GridSearchCV amb cross-validation i valida el millor model amb cross-validation final.
    """
    for model, param_grid, title in models:
        print(f"\n[INFO] Aplicant GridSearchCV al model {title}...")

        grid_search = GridSearchCV(
            model, param_grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=2
        )
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"[RESULTS] {title}: Millors paràmetres -> {best_params}")
        print(f"[RESULTS] {title}: Millor accuracy -> {best_score:.4f}")

        print(f"\n[INFO] Realitzant cross-validation amb els millors paràmetres per al model {title}...")
        cv_scores = cross_validation(best_model, X, y)

        resultats[title] = {
            "best_params": best_params,
            "grid_search_best_score": best_score,
            "cross_val_scores": cv_scores.tolist(),
            "cross_val_mean": np.mean(cv_scores)
        }


def grid_search_only(X_train, y_train, resultats, models):
    """
    Aplica només GridSearchCV al classificador amb augmentació.
    """
    for model, param_grid, title in models:
        print(f"\n[INFO] Aplicant GridSearchCV al model {title}...")

        grid_search = GridSearchCV(
            model, param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=2
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"[RESULTS] {title}: Millors paràmetres -> {best_params}")
        print(f"[RESULTS] {title}: Millor accuracy -> {best_score:.4f}")

        resultats[title] = {
            "best_params": best_params,
            "grid_search_best_score": best_score
        }


def guardar_resultats_a_json(resultats, nom_fitxer="bestModels_gridSearch_CV.json"):
    """
    Guarda els resultats en un fitxer JSON.
    """
    current_dir = Path(r"C:/Users/carlo/Desktop/uni/AC/Projecte AC/ACproject-14-grup/espectogrames/jasond")
    fitxer_json = current_dir / nom_fitxer

    with open(fitxer_json, "w") as fitxer:
        json.dump(resultats, fitxer, indent=4)
    print(f"[SUCCESS] Resultats guardats a {fitxer_json}")


if __name__ == "__main__":
    base_dir = "ACproject-14-grup/datasets/Data1/images_original"
    resultats_original = {}
    resultats_augmented = {}

    print("[INFO] Processant dades...")
    data, labels = [], []
    processament(data, labels, base_dir)

    data = pd.DataFrame(data)
    data["label"] = labels
    data, label_encoder = codificar_label(data)
    X, y = definirXY_normalitzar(data)

    print("\n[INFO] Aplicant GridSearchCV i cross-validation amb el dataset original...")
    
    models = [
    (LogisticRegression(), 
     {"C": [0.01, 0.1, 1, 10, 100, 1000],
      "solver": ["lbfgs", "newton-cg", "saga"],
      "penalty": ["l2", "elasticnet"],
      "max_iter": [100, 500, 1000]},
     "Logistic Regression"),
    
    (RandomForestClassifier(), 
     {"n_estimators": [800, 1000, 1500], 
      "max_depth": [18, 20, 30, 35], 
      "max_features": ["sqrt", "log2", None],
      "min_samples_split": [2, 3], 
      "min_samples_leaf": [1, 2, 3, 4]}, 
     "Random Forest")]

    
    grid_search_with_cv_and_final_validation(X, y, resultats_original, models)

    print("\n[INFO] Dividint dataset en conjunt train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)

    print("\n[INFO] Aplicant augmentació al conjunt d'entrenament...")
    augmented_data, augmented_labels = [], []
    for i in range(len(X_train)):
        img = X_train.iloc[i].values.reshape(128, 128)
        genre = y_train.iloc[i]
        augmented_images = augment_image(img)
        for aug_img in augmented_images:
            augmented_data.append(aug_img.flatten())
            augmented_labels.append(genre)

    X_train_augmented = pd.DataFrame(augmented_data)
    y_train_augmented = pd.Series(augmented_labels)

    print("\n[INFO] Aplicant GridSearchCV al dataset amb augmentació...")
    grid_search_only(X_train_augmented, y_train_augmented, resultats_augmented, models)

    print("\n[INFO] Guardant els resultats...")
    guardar_resultats_a_json(resultats_original, "resultats_original.json")
    guardar_resultats_a_json(resultats_augmented, "resultats_augmented.json")
    print("[SUCCESS] Resultats guardats correctament!")
