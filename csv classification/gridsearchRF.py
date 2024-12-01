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



# Definir els hiperparàmetres per a la cerca
param_grid = {
    'n_estimators': [100, 500],  # Redueix les opcions
    'max_depth': [10, 20],  # Redueix el rang de profunditat
    'min_samples_split': [2, 5],  # Pocs valors per dividir nodes
    'min_samples_leaf': [1, 2],  # Pocs valors per les fulles
    'max_features': ['sqrt']  # Limita les opcions de característiques
}

# Llista de models a avaluar:
rf = RandomForestClassifier(random_state = 42)
    # (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0), "Gradient Boosting"),
    # (XGBClassifier(n_estimators=1000, learning_rate=0.05), "Cross Gradient Booster"),

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',  # Es pot canviar a 'roc_auc' o altres mètriques
    cv=3,  # Validació creuada amb 5 particions
    verbose=2,
    n_jobs=-1  # Utilitzar tots els nuclis disponibles per accelerar
)

data30s = codificar_label(data30s)  # Codificar etiquetes
X, y = definirXY_normalitzar(data30s)  # Definir X i y
X_train, X_test, y_train, y_test = divisio_dades(X, y)

param_distributions = {
    'n_estimators': randint(500, 2000),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,  # Augmenta les iteracions per més combinacions
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("\nMillors hiperparàmetres trobats:")
print(random_search.best_params_)
print(f"Millor precisió aconseguida: {random_search.best_score_:.4f}")

# # Executar GridSearchCV
# print("Iniciant cerca d'hiperparàmetres...")
# grid_search.fit(X_train, y_train)

# # Resultats
# print("\nMillors hiperparàmetres trobats:")
# print(grid_search.best_params_)
# print(f"Millor precisió aconseguida: {grid_search.best_score_:.4f}")

# # Utilitzar el millor model per fer prediccions
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test)

# # Avaluar el model final
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nAccuracy del millor model en el conjunt de test: {accuracy:.4f}")