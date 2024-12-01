import os
from pathlib import Path

import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display as ipd
from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier

def codificar_label(data):
    label_encoder = preprocessing.LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data

def definirXY_normalitzar(data):
    X = data.drop(['label','filename'],axis=1) #treiem label(vaalor a predir) i filename (redundant)
    y = data['label'] #variable independent (a predir)
    columnes = X.columns
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X) #escalem 
    X = pd.DataFrame(np_scaled, columns=columnes)#nou dataset sense label i filename
    return X, y

 
def divisio_dades(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=111)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape} (test_size={test_size})") #mostrem dimensions
    return X_train, X_test, y_train, y_test



def model_assess(model, X_train, X_test, y_train, y_test, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n') #calcular accuracy

def model_assess_to_json(model, X_train, X_test, y_train, y_test, title, resultats, dataset):
    """
    Avaluar un model amb diverses mètriques i guardar els resultats en un diccionari.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Calcular mètriques per al test
    accuracy_test = round(accuracy_score(y_test, preds), 5)
    precision_test = round(precision_score(y_test, preds, average="weighted", zero_division=0), 5)
    recall_test = round(recall_score(y_test, preds, average="weighted", zero_division=0), 5)
    f1_test = round(f1_score(y_test, preds, average="weighted", zero_division=0), 5)
    roc_auc_test = round(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"), 5) if hasattr(model, "predict_proba") else None

    # Calcular mètriques per al train
    preds_train = model.predict(X_train)
    accuracy_train = round(accuracy_score(y_train, preds_train), 5)
    f1_train = round(f1_score(y_train, preds_train, average="weighted", zero_division=0), 5)

    # Calcular generalization gap
    accuracy_gap = round(accuracy_train - accuracy_test, 5)
    f1_gap = round(f1_train - f1_test, 5)

    # Assegurar que existeix la clau del model
    if title not in resultats[dataset]:
        resultats[dataset][title] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "roc_auc": [],
            "accuracy_gap": [],
            "f1_gap": []
        }

    # Afegir mètriques al diccionari
    resultats[dataset][title]["accuracy"].append(accuracy_test)
    resultats[dataset][title]["precision"].append(precision_test)
    resultats[dataset][title]["recall"].append(recall_test)
    resultats[dataset][title]["f1_score"].append(f1_test)
    if roc_auc_test is not None:
        resultats[dataset][title]["roc_auc"].append(roc_auc_test)
    resultats[dataset][title]["accuracy_gap"].append(accuracy_gap)
    resultats[dataset][title]["f1_gap"].append(f1_gap)



def guardar_resultats_a_json(resultats, nom_fitxer="resultats.json"):
    """
    Guarda els resultats en un fitxer JSON.
    """
    with open(nom_fitxer, "w") as fitxer:
        json.dump(resultats, fitxer, indent=4)
    print(f"Resultats guardats a {nom_fitxer}")


if __name__ == "__main__":

    current_dir = Path(__file__).parent

    # Construir el camí als csv
    cami_csv_3s = current_dir.parent / "datasets" / "Data1" / "features_3_sec.csv"
    cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

    data3s = pd.read_csv(cami_csv_3s)
    data30s = pd.read_csv(cami_csv_30s)

    # Diferents mides de train i test
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Llista de models a avaluar:
    models = [
        (GaussianNB(), "Gaussian Naive Bayes"),
        (BernoulliNB(), "Bernoulli Naive Bayes"),
        (MultinomialNB(), "Multinomial Naive Bayes"),
        (SVC(decision_function_shape="ovo"), "Support Vector Machine"),
        (KNeighborsClassifier(n_neighbors=19), "K-Nearest Neighbors"),
        (DecisionTreeClassifier(), "Decision Trees"),
        (RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0), "Random Forest"),
        (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0), "Gradient Boosting"),
        (XGBClassifier(n_estimators=1000, learning_rate=0.05), "Cross Gradient Booster"),
        (XGBRFClassifier(objective= 'multi:softmax'),"Cross Gradient Booster (Random Forest)" )

    ]

    resultats = {"3 seconds": {}, "30 seconds": {}}

    # Avaluació de cada dataset
    for data, tipus in [(data3s, "3 seconds"), (data30s, "30 seconds")]:  # Diferents datasets
        print(f"\n### Avaluant {tipus} data ###")
        data = codificar_label(data)
        X, y = definirXY_normalitzar(data)

        for test_size in test_sizes:  # Diferents mides
            print(f"\n--- Test size: {test_size} ---")
            X_train, X_test, y_train, y_test = divisio_dades(X, y, test_size=test_size)

            for model, title in models:
                print(f"\nModel: {title}")
                model_assess_to_json(model, X_train, X_test, y_train, y_test, title, resultats, dataset=tipus)


    # Guarda els resultats al fitxer JSON
    guardar_resultats_a_json(resultats)