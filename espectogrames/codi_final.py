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
from multiprocessing import Process

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
                    img_resized = cv2.resize(img, img_size) / 255.0
                    data.append(img_resized.flatten())
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

# Entrenar el model
def entrenar_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def model_assess_to_json_timer(model, X_train, X_test, y_train, y_test, title, resultats, timeout=360):
    """
    Avaluar un model amb diverses mètriques i interrompre si supera el temps límit (timeout).
    """
    if title not in resultats:
        resultats[title] = {}

    process = Process(target=entrenar_model, args=(model, X_train, y_train))
    process.start()

    # Supervisar el temps
    process.join(timeout)
    if process.is_alive():
        print(f"Model {title} interromput després de {timeout / 60:.1f} minuts.")
        process.terminate()
        process.join()
        resultats[title]["Error"] = f"Entrenament interromput després de {timeout / 60:.1f} minuts."
        return  # Surt de la funció si l'entrenament es cancel·la

    # Continuar amb l'avaluació si el model s'ha entrenat correctament
    start_predict = time.time()
    preds = model.predict(X_test)
    predict_time = round(time.time() - start_predict, 5)

    # Calcular mètriques per al test
    accuracy_test = round(accuracy_score(y_test, preds), 5)
    precision_test = round(precision_score(y_test, preds, average="weighted", zero_division=0), 5)
    recall_test = round(recall_score(y_test, preds, average="weighted", zero_division=0), 5)
    f1_test = round(f1_score(y_test, preds, average="weighted", zero_division=0), 5)
    
    if len(np.unique(y_test)) > 1 and hasattr(model, "predict_proba"):
        roc_auc_test = round(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"), 5)
    else:
        roc_auc_test = None

    # Afegir mètriques al diccionari
    resultats[title]["accuracy"] = accuracy_test
    resultats[title]["precision"] = precision_test
    resultats[title]["recall"] = recall_test
    resultats[title]["f1_score"] = f1_test
    if roc_auc_test is not None:
        resultats[title]["roc_auc"] = roc_auc_test
    resultats[title]["temps_predict"] = predict_time

def guardar_resultats_a_json(resultats, nom_fitxer="resultats_Comp2_totsmodels_timer.json"):
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
   
    data, labels = [], []
    processament(data, labels, img_size=(128,128))
    
    data = pd.DataFrame(data)
    data["label"] = labels

    data, label_encoder = codificar_label(data)
    X, y = definirXY_normalitzar(data)

    """
    # Cross-Validation sobre el dataset original (sense augmentació)
    print("\nRealitzant Cross-Validation sobre el dataset original...")
    pipeline_original = Pipeline([('pca', PCA(n_components=100)), ('rf', RandomForestClassifier(n_estimators=1000, max_depth=20, max_features="sqrt", min_samples_split=2, min_samples_leaf=1, random_state= 32))])    
    cv_scores = cross_validation(pipeline_original, X, y)

    resultats["Random Forest CV sense augmentació"]["cross_val_scores"] = cv_scores.tolist()
    resultats["Random Forest CV sense augmentació"]["cross_val_mean"] = np.mean(cv_scores)
    """    

    print("\nDividint dataset en conjunt train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)
    
    # Aplicar augmentació només al conjunt d’entrenament
    print("\nAplicant augmentació al conjunt d'entrenament...")
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
    

    # Entrenar el model Random Forest
    print("\nEntrenant els models...")
    for model, title in models:
        print(f"\nModel: {title}")
        pipeline_augmentat = Pipeline([('pca', PCA(n_components=100)), ('model', model)])
       
        model_assess_to_json_timer(pipeline_augmentat, X_train_augmented, X_test, y_train_augmented, y_test, title, resultats)

    # Guarda els resultats al fitxer JSON
    guardar_resultats_a_json(resultats)