from pathlib import Path
from matplotlib import pyplot as plt
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
    cami_audio = current_dir.parent / "audio classification" / "audio_features_prova2.csv"
    # cami_csv_30s = current_dir.parent / "datasets" / "Data1" / "features_30_sec.csv"

    dataaudio = pd.read_csv(cami_audio)
    # data30s = pd.read_csv(cami_csv_30s)

    # Encode label
    data = codificar_label(dataaudio)
    
    # Define X and y and normalize
    X, y = definirXY_normalitzar(data)

    # Split data
    X_train, X_test, y_train, y_test = divisio_dades(X, y)

    model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.15, max_depth=10, min_samples_leaf=12, min_samples_split=2)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    confusion_matr = confusion_matrix(y_test, preds) #normalize = 'true'
    plt.figure(figsize = (16, 9))
    sns.heatmap(confusion_matr, cmap="Blues", annot=True, 
                xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
            yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);
    plt.savefig("conf matrix")
    plt.show()