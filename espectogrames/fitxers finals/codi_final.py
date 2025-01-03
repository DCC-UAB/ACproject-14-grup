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

def augment_image(image):
    flipped = np.fliplr(image)
    noisy = np.clip(image + np.random.normal(0, 0.02, image.shape), 0, 1)  
    brighter = np.clip(image * 1.2, 0, 1)
    darker = np.clip(image * 0.8, 0, 1)
    return [image, flipped, noisy, brighter, darker]

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


def model_assess_to_json(model, X_train, X_test, y_train, y_test, title, resultats):
    """
    Avaluar un model amb diverses mètriques i guardar els resultats en un diccionari.
    """
    if title not in resultats:
        resultats[title] = {}

    start_total = time.time()

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_train, 5)

    start_predict = time.time()
    preds = model.predict(X_test)
    predict_time = round(time.time() - start_predict, 5)

    total_time = round(time.time() - start_total, 5)

    # Calcular mètriques per al test
    accuracy_test = round(accuracy_score(y_test, preds), 5)
    precision_test = round(precision_score(y_test, preds, average="weighted", zero_division=0), 5)
    recall_test = round(recall_score(y_test, preds, average="weighted", zero_division=0), 5)
    f1_test = round(f1_score(y_test, preds, average="weighted", zero_division=0), 5)
    
    if len(np.unique(y_test)) > 1 and hasattr(model, "predict_proba"):
        roc_auc_test = round(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"), 5)
    else:
        roc_auc_test = None
    
    # Calcular mètriques per al train
    preds_train = model.predict(X_train)
    accuracy_train = round(accuracy_score(y_train, preds_train), 5)
    f1_train = round(f1_score(y_train, preds_train, average="weighted", zero_division=0), 5)

    # Calcular generalization gap
    accuracy_gap = round(accuracy_train - accuracy_test, 5)
    f1_gap = round(f1_train - f1_test, 5)


    # Afegir mètriques al diccionari
    resultats[title]["accuracy"]=accuracy_test
    resultats[title]["precision"]=precision_test
    resultats[title]["recall"]=recall_test
    resultats[title]["f1_score"]=f1_test
    if roc_auc_test is not None:
        resultats[title]["roc_auc"]=roc_auc_test
    resultats[title]["accuracy_gap"]=accuracy_gap
    resultats[title]["f1_gap"]=f1_gap
    resultats[title]["temps_train"]=train_time
    resultats[title]["temps_predict"]=predict_time
    resultats[title]["temps_total"]=total_time


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


def guardar_resultats_a_json(resultats, nom_fitxer="totsmodels_caracteristiquesAudio_plus.json"):
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

    models = [(BernoulliNB(), "Naive Bayes (BernoulliNB)"),
              (GaussianNB(), "Naive Bayes (GaussianNB)"),
              (MultinomialNB(), "Naive Bayes (MultinomialNB)"),
              (LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression"),
              (KNeighborsClassifier(n_neighbors=7), "K-Nearest Neighbors"),
              (DecisionTreeClassifier(random_state=42), "Decision Tree"),
              (SVC(kernel="rbf", probability=True, random_state=42), "Support Vector Machine (SVM)"),
              (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
              (GradientBoostingClassifier(random_state=42), "Gradient Boosting"),
              (XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42), "XGBoost (XGB)"),
              (XGBRFClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42), "XGBoost (XGBRF)")] 
   
    data, labels = [], []
    print("[INFO] Processant dades i extraient característiques d'audio...")
    processament(data, labels, img_size=(128,128))
    
    print("[INFO] Codificant etiquetes i normalitzant dades...")
    data = pd.DataFrame(data)
    data["label"] = labels

    data, label_encoder = codificar_label(data)
    X, y = definirXY_normalitzar(data)

    """
    # Cross-Validation sobre el dataset original (sense augmentació)
    print("\nRealitzant Cross-Validation sobre el dataset original...")
    for model, title in models:
        print(f"\nModel: {title}")
    
        cv_scores = cross_validation(model, X, y)
        resultats["Random Forest CV sense augmentació"]["cross_val_scores"] = cv_scores.tolist()
        resultats["Random Forest CV sense augmentació"]["cross_val_mean"] = np.mean(cv_scores)
   """

    print("\n[INFO] Dividint dataset en conjunt train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)

    """
    print("\n[INFO] Aplicant augmentació només al conjunt d'entrenament...")
    augmented_data, augmented_labels = [], []
    for i in range(len(X_train)):
        img = X_train.iloc[i].values.reshape(128, 128)  # Reconstrueix la imatge 2D
        genre = y_train.iloc[i]
        augmented_images = augment_image(img)  
        for aug_img in augmented_images:
            augmented_data.append(aug_img.flatten())
            augmented_labels.append(genre)

    X_train_augmented = pd.DataFrame(augmented_data)
    y_train_augmented = pd.Series(augmented_labels)
    """

    print("\n[INFO] Entrenant els models...")
    for model, title in models:
        print(f"\n[INFO] Començant l'entrenament per al model {title}...")
        model_assess_to_json_timer(model, X_train, X_test, y_train, y_test, title, resultats)

    print("\n[INFO] Guardant els resultats al fitxer JSON...")
    guardar_resultats_a_json(resultats)
    print("[SUCCESS] Resultats guardats correctament!")