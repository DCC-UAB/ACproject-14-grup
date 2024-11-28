import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import librosa
import librosa.display
import IPython.display as ipd
import warnings
import os
from sklearn import preprocessing
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


current_dir = Path(__file__).parent

# Construir el camí als csv
cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

data30s = pd.read_csv(cami_csv_30s)

X = data30s.drop(['label','filename'],axis=1)
y = data30s['label'] 

columnes = X.columns
min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns=columnes)

# Divisió inicial: test (20%) i restant(train + validacio) (80%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=111)

# Divisió del restant en entrenament (70% del total) i validació (10% del total)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=111)  # 0.25 x 0.8 = 0.2 --> midea validacio

# Mostrem les dimensions
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")