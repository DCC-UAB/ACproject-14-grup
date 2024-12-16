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
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import threading
import librosa
import librosa.display
from skimage.feature import hog

def processament(data, labels, img_size=(128,128)):
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        if os.path.isdir(genre_path):
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    features, _ = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                    data.append(features)
                    labels.append(genre)
            


def codificar_label(data):
    label_encoder = preprocessing.LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data, label_encoder

def definirXY_normalitzar(data):
    X = data.drop(['label'],axis=1)
    y = data['label'] #variable independent (a predir)
    columnes = X.columns
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X) #escalem 
    X = pd.DataFrame(np_scaled, columns=columnes)#nou dataset sense label i filename
    return X, y

def cross_validation(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    print(f"Accuracy mitjà amb cross-validation: {np.mean(scores):.4f}")
    return scores


def guardar_resultats_a_json(resultats, nom_fitxer="bestModels_GS+CV_hog.json"):
    """
    Guarda els resultats en un fitxer JSON.
    """
    current_dir = Path(r"C:/Users/carlo/Desktop/uni/AC/Projecte AC/ACproject-14-grup/espectogrames/jasond")
    
    fitxer_json = current_dir / nom_fitxer  # Camí complet al fitxer JSON

    with open(nom_fitxer, "w") as fitxer:
        json.dump(resultats, fitxer, indent=4)
    print(f"Resultats guardats a {fitxer_json}")


if __name__ == "__main__":
    base_dir = "ACproject-14-grup/datasets/Data1/images_original"
    resultats = {}   
 
    print("[INFO] Processant dades i extraient característiques HOG...")
    data, labels = [], []
    processament(data, labels, img_size=(128,128))
    data = pd.DataFrame(data)
    data["label"] = labels

    print("[INFO] Codificant etiquetes i normalitzant dades...")
    data, label_encoder = codificar_label(data)
    X, y = definirXY_normalitzar(data)

    print("\n[INFO] Dividint dataset en conjunt train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)

    param_grid_lr = {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs", "saga"]}
    param_grid_svm = {"C": [0.1, 1, 5, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto", 0.01]}
    param_grid_xgb = {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2], "max_depth": [3, 6, 9, 12]}

    models = [
        (LogisticRegression(), param_grid_lr, "Logistic Regression"),
        (SVC(probability=True), param_grid_svm, "Support Vector Machine (SVM)"),
        (XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42), param_grid_xgb, "XGBoost")
    ]

    for model, param_grid, title in models:
        print(f"\n[INFO] Aplicant GridSearchCV al model {title}...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        print(f"[RESULTS] {title}: Millors paràmetres -> {grid_search.best_params_}")
        print(f"[RESULTS] {title}: Millor accuracy -> {grid_search.best_score_:.4f}")

        resultats[title] = {
            "best_params": grid_search.best_params_,
            "best_accuracy": grid_search.best_score_
        }

        print(f"\n[INFO] Realitzant cross-validation amb els millors paràmetres per al model {title}...")
        best_model = grid_search.best_estimator_
        cv_scores = cross_validation(best_model, X, y)
        resultats[title]["cross_val_scores"] = cv_scores.tolist()
        resultats[title]["cross_val_mean"] = np.mean(cv_scores)

        print(f"\n[INFO] Avaluant el model {title} en conjunt de test...")
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"[RESULTS] {title}: Accuracy en test -> {test_accuracy:.4f}")
        resultats[title]["test_accuracy"] = test_accuracy

    guardar_resultats_a_json(resultats)