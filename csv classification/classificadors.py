import os
from pathlib import Path

import librosa
import librosa.display
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
    X = data.drop(['label','filename'],axis=1) #treiem label(vaalor a predir) i filename (redundant)
    y = data['label'] #variable independent (a predir)
    columnes = X.columns
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X) #escalem 
    X = pd.DataFrame(np_scaled, columns=columnes)#nou dataset sense label i filename
    return X, y

 
def divisio_dades(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    # Mostrem les dimensions
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def model_assess(model, X_train, X_test, y_train, y_test, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n') #calcular accuracy


data = codificar_label(data3s)
X, y = definirXY_normalitzar(data)
X_train, X_test, y_train, y_test = divisio_dades(X, y)

# Llista de models a avaluar:
models = [
    (GaussianNB(), "Gaussian Naive Bayes"),
    (BernoulliNB(), "Bernoulli Naive Bayes"),
    (MultinomialNB(), "Multinomial Naive Bayes"),
    (SVC(decision_function_shape="ovo"), "Support Vector Machine"),
    (KNeighborsClassifier(n_neighbors=19), "K-Nearest Neighbors"),
    (DecisionTreeClassifier(), "Decision Trees"),
    (RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0), "Random Forest"),
    (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0), "Gradient Boosting")
]

# Avaluació de cada model:
for model, title in models:
    model_assess(model, X_train, X_test, y_train, y_test, title)