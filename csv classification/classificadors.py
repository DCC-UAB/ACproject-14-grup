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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import time  # Per mesurar el temps d'execució

current_dir = Path(__file__).parent

# Construir el camí als csv
cami_csv_3s = current_dir.parent / "datasets" / "dades_sense_outliers_3s.csv"
cami_csv_30s = current_dir.parent / "datasets" / "dades_sense_outliers_30s.csv"

data3s = pd.read_csv(cami_csv_3s)
data30s = pd.read_csv(cami_csv_30s)

def codificar_label(data):
    label_encoder = preprocessing.LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data

def definirXY_normalitzar(data):
    X = data.drop(['label', 'filename'], axis=1)  # Treure label (valor a predir) i filename (redundant)
    y = data['label']  # Variable independent (a predir)
    columnes = X.columns
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)  # Escalem
    X = pd.DataFrame(np_scaled, columns=columnes)  # Nou dataset sense label ni filename
    return X, y

def divisio_dades(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=111)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape} (test_size={test_size})")  # Mostrem dimensions
    return X_train, X_test, y_train, y_test

def model_assess_to_json(model, X_train, X_test, y_train, y_test, title, resultats, dataset):
    """
    Avaluar un model i guardar l'accuracy al diccionari de resultats.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, preds), 5)
    
    # Assegura't que la clau del model existeix al dataset
    if title not in resultats[dataset]:
        resultats[dataset][title] = []
    
    # Afegeix l'accuracy
    resultats[dataset][title].append(accuracy)

def guardar_resultats_a_json(resultats, nom_fitxer="resultats.json"):
    """
    Guarda els resultats en un fitxer JSON.
    """
    with open(nom_fitxer, "w") as fitxer:
        json.dump(resultats, fitxer, indent=4)
    print(f"Resultats guardats a {nom_fitxer}")


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
    (XGBRFClassifier(objective='multi:softmax'), "Cross Gradient Booster (Random Forest)")
]

resultats = {"3 seconds": {}, "30 seconds": {}}

# Mesura del temps
start_total = time.time()

# Avaluació de cada dataset
for data, tipus in [(data3s, "3 seconds"), (data30s, "30 seconds")]:  # Diferents datasets
    print(f"\n### Avaluant {tipus} data ###")
    start_dataset = time.time()  # Temps inicial per al dataset

    data = codificar_label(data)
    X, y = definirXY_normalitzar(data)

    for test_size in test_sizes:  # Diferents mides
        print(f"\n--- Test size: {test_size} ---")
        start_test_size = time.time()  # Temps inicial per al test size

        X_train, X_test, y_train, y_test = divisio_dades(X, y, test_size=test_size)

        for model, title in models:
            start_model = time.time()  # Temps inicial per al model
            print(f"\nModel: {title}")
            model_assess_to_json(model, X_train, X_test, y_train, y_test, title, resultats, dataset=tipus)
            end_model = time.time()
            print(f"Temps per {title} amb test size {test_size}: {end_model - start_model:.2f} segons")

        end_test_size = time.time()
        print(f"Temps total per test size {test_size}: {end_test_size - start_test_size:.2f} segons")

    end_dataset = time.time()
    print(f"Temps total per {tipus} data: {end_dataset - start_dataset:.2f} segons")

# Guarda els resultats al fitxer JSON
guardar_resultats_a_json(resultats)

end_total = time.time()
print(f"\nTemps total d'execució: {end_total - start_total:.2f} segons")
