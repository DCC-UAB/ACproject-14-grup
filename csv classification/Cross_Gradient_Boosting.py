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
from scipy.stats import randint, uniform

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


def guardar_resultats_a_json(resultats, nom_fitxer="resultats_Cross_Gradient_Boosting_best_hyperparametresP2.json"):
    """
    Guarda els resultats en un fitxer JSON.
    """
    current_dir = Path(__file__).parent.resolve()  # Directori actual del script
    fitxer_json = current_dir / nom_fitxer  # Camí complet al fitxer JSON

    with open(fitxer_json, "w") as fitxer:
        json.dump(resultats, fitxer, indent=4)
    print(f"Resultats guardats a {fitxer_json}")

def grid_search_xgb(X_train, y_train):
    """
    Realitza una cerca d'hiperparàmetres utilitzant GridSearchCV per a XGBClassifier.
    """
    #PROVA 1 
    #{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500, 'subsample': 0.8}
    # param_grid = {
    #     'n_estimators': [100, 200, 500],
    #     'max_depth': [3, 6, 10],
    #     'learning_rate': [0.01, 0.1, 0.3],
    #     'subsample': [0.8, 1.0],
    #     'colsample_bytree': [0.8, 1.0],
    # }
    
    #PROVA 2
    #DONAT ELS MILLORS PARAMETRES DE LA PROVA 1 SEGUIM FENT CERCA PER VALORS PROEPERS ALS OBTINGUTS 
    
    param_grid = {
    'learning_rate': [0.05,  0.1, 0.15],
    'max_depth': [ 5, 6, 7],
    'n_estimators': [400, 500, 600],
    'subsample': [0.75, 0.8, 0.85],
    'colsample_bytree': [0.75, 0.8, 0.85]
}

   
    xgb = XGBClassifier(tree_method='hist', random_state=42)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='f1_weighted', verbose=2, n_jobs=-1)

   
    grid_search.fit(X_train, y_train)
    
    print("\nMillors hiperparàmetres trobats:")
    print(grid_search.best_params_)
    print("\nMillor puntuació obtinguda (F1-weighted):")
    print(grid_search.best_score_)

    return grid_search.best_estimator_

def random_seach_hyperparameters(X_train, y_train):
    
    xgb = XGBClassifier(tree_method='hist', random_state=42)
     # Definir el grid d'hiperparàmetres
    param_distributions = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 15),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'gamma': uniform(0, 5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(1, 10)
    }
     # Configura RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=2000,  # Nombre de combinacions aleatòries
        scoring='accuracy',  # Pots canviar la mètrica
        cv=5,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    
    random_search.fit(X_train, y_train)

    # Resultats
    print("\nMillors hiperparàmetres trobats amb RandomizedSearchCV:")
    print(random_search.best_params_)

    return random_search.best_estimator_
    


if __name__ == "__main__":

    current_dir = Path(__file__).parent

    # Construir el camí als csv
    cami_csv_3s = current_dir.parent / "csv classification" / "features_3_sec_top20.csv"

    data3s = pd.read_csv(cami_csv_3s)

    resultats = {}

    # Model Random Forest
    # xgb = XGBClassifier(tree_method='hist', random_state=42)
    # model_title = "Cross Gradient Boosting"

    # Avaluació de cada dataset
    for data, tipus in [(data3s, "3 seconds")]:  # Diferents datasets
        print(f"\n### Avaluant {tipus} data ###")
        data = codificar_label(data)
        X, y = definirXY_normalitzar(data)

        X_train, X_test, y_train, y_test = divisio_dades(X, y, test_size=0.2)
        # millor_model = grid_search_xgb(X_train, y_train)
        millor_model = random_seach_hyperparameters(X_train, y_train)
        # millor_model = XGBClassifier(
        #     colsample_bytree=0.85,
        #     learning_rate=0.15,
        #     max_depth=7,
        #     n_estimators=400,
        #     subsample=0.8,
        #     tree_method='hist',
        #     random_state=42
        # )

        # Avaluar i guardar resultats del model optimitzat
        # model_assess_to_json(millor_model, X_train, X_test, y_train, y_test, "Optimized XGBClassifier", resultats, dataset=tipus)

    # Guarda els resultats al fitxer JSON
    # guardar_resultats_a_json(resultats)

