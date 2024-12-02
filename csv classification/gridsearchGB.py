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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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

# Definim el grid d'hiperparàmetres
param_grid = {
    'n_estimators': [100, 200, 500],  # Nombre d'arbres
    'learning_rate': [0.01, 0.1, 0.2],  # Ritme d'aprenentatge
    'max_depth': [3, 5, 7],  # Profunditat màxima de cada arbre
    'min_samples_split': [2, 5, 10],  # Mida mínima per dividir nodes
    'min_samples_leaf': [1, 2, 4],  # Mida mínima de les fulles
    'subsample': [0.8, 1.0]  # Percentatge de mostres utilitzades per arbre
}

# Inicialitzem GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)

# GridSearchCV per a GradientBoostingClassifier
grid_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    scoring='accuracy',  # Pots canviar per roc_auc o una altra mètrica
    cv=5,  # Validació creuada amb 5 particions
    verbose=2,
    n_jobs=-1
)

data30s = codificar_label(data30s)  # Codificar etiquetes
X, y = definirXY_normalitzar(data30s)  # Definir X i y
X_train, X_test, y_train, y_test = divisio_dades(X, y)

# Entrenem GridSearchCV
print("Iniciant cerca d'hiperparàmetres per GradientBoostingClassifier...")
grid_search.fit(X_train, y_train)

# Resultats
print("\nMillors hiperparàmetres trobats per GradientBoostingClassifier:")
print(grid_search.best_params_)
print(f"Millor precisió aconseguida: {grid_search.best_score_:.4f}")

# Model final amb els millors hiperparàmetres
best_gb = grid_search.best_estimator_
y_pred = best_gb.predict(X_test)

# Avaluem el rendiment del model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy del millor GradientBoostingClassifier en el conjunt de test: {accuracy:.4f}")