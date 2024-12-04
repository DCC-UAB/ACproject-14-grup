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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

current_dir = Path(__file__).parent

# Construir el camí als csv
cami_csv_3s = current_dir.parent / "datasets" / "Data1" / "features_3_sec.csv"
cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

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

def divisio_dades(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=111)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape} (test_size={test_size})")  # Mostrem dimensions
    return X_train, X_test, y_train, y_test

def model_assess_to_json(model, X_train, X_test, y_train, y_test, title, resultats, dataset):
    """
    Avaluar Random Forest amb diverses mètriques i guardar els resultats en un diccionari.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Calcular mètriques per al test
    accuracy_test = round(accuracy_score(y_test, preds), 5)
    precision_test = round(precision_score(y_test, preds, average="weighted", zero_division=0), 5)
    recall_test = round(recall_score(y_test, preds, average="weighted", zero_division=0), 5)
    f1_test = round(f1_score(y_test, preds, average="weighted", zero_division=0), 5)
    roc_auc_test = round(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"), 5)

    # Calcular mètriques per al train
    preds_train = model.predict(X_train)
    accuracy_train = round(accuracy_score(y_train, preds_train), 5)
    f1_train = round(f1_score(y_train, preds_train, average="weighted", zero_division=0), 5)

    # Calcular generalization gap
    accuracy_gap = round(accuracy_train - accuracy_test, 5)
    f1_gap = round(f1_train - f1_test, 5)

    # Afegir mètriques al diccionari
    resultats[dataset] = {
        title: {
            "accuracy": accuracy_test,
            "precision": precision_test,
            "recall": recall_test,
            "f1_score": f1_test,
            "roc_auc": roc_auc_test,
            "accuracy_gap": accuracy_gap,
            "f1_gap": f1_gap
        }
    }


def guardar_resultats_a_json(resultats, nom_fitxer="resultats_random_forest_clean_dataset.json"):
    """
    Guarda els resultats en un fitxer JSON.
    """
    current_dir = Path(__file__).parent.resolve()  # Directori actual del script
    fitxer_json = current_dir / nom_fitxer  # Camí complet al fitxer JSON

    with open(fitxer_json, "w") as fitxer:
        json.dump(resultats, fitxer, indent=4)
    print(f"Resultats guardats a {fitxer_json}")


if __name__ == "__main__":

    current_dir = Path(__file__).parent

    # Construir el camí als csv
    cami_csv_3s = current_dir.parent / "csv classification" / "features_3_sec_top20.csv"

    data3s = pd.read_csv(cami_csv_3s)

    resultats = {}

    # Model Random Forest
    random_forest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
    model_title = "Random Forest"

    # Avaluació de cada dataset
    for data, tipus in [(data3s, "3 seconds")]:  # Diferents datasets
        print(f"\n### Avaluant {tipus} data ###")
        data = codificar_label(data)
        X, y = definirXY_normalitzar(data)

        X_train, X_test, y_train, y_test = divisio_dades(X, y, test_size=0.2)

        # Avaluar i guardar resultats
        model_assess_to_json(random_forest, X_train, X_test, y_train, y_test, model_title, resultats, dataset=tipus)

    # Guarda els resultats al fitxer JSON
    guardar_resultats_a_json(resultats)

