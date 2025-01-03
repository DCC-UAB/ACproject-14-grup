from pathlib import Path
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


if __name__ == "__main__":

    current_dir = Path(__file__).parent

    # Construir el camí als csv
    cami_csv_3s = current_dir.parent / "csv classification" / "features_3_sec_top20_XGBoost.csv"
    # cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

    data3s = pd.read_csv(cami_csv_3s)
    # data30s = pd.read_csv(cami_csv_30s)

    # Encode label
    data = codificar_label(data3s)

    # Define X and y and normalize
    X, y = definirXY_normalitzar(data)

    # Split data
    X_train, X_test, y_train, y_test = divisio_dades(X, y)

    # Create model
    millor_model = XGBClassifier(
        colsample_bytree=0.85,
        learning_rate=0.075,
        max_depth=5,
        n_estimators=1500,
        subsample=0.8,
        tree_method='hist',
        random_state=42
    )

    # Cross validation
    kfold = KFold(n_splits=10, random_state=111, shuffle=True)

    results = cross_val_score(millor_model, X, y, cv=kfold)

    print("Best Accuracy: %.2f%%" % (results.max()*100), "Mean Accuracy: %.2f%%" % (results.mean()*100), "Std: %.2f%%" % (results.std()*100))