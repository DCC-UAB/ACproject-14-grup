from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

def codificar_label(data):
    label_encoder = preprocessing.LabelEncoder()
    data['genre'] = label_encoder.fit_transform(data['genre'])
    return data

def definirXY_normalitzar(data):
    X = data.drop(['genre', 'segment'], axis=1)  # Treure label (valor a predir) i filename (redundant)
    y = data['genre']  # Variable independent (a predir)
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
    current_dir = Path(__file__).parent.resolve()  # Directori actual del script
    
    # Camí absolut al fitxer CSV
    cami_csv = current_dir / "audio_features_prova2.csv"  # Ajusta el camí per apuntar al fitxer directament

    data = pd.read_csv(cami_csv)
    # data30s = pd.read_csv(cami_csv_30s)

    # Encode label
    data = codificar_label(data)

    # Define X and y and normalize
    X, y = definirXY_normalitzar(data)

    # Split data
    X_train, X_test, y_train, y_test = divisio_dades(X, y)

    # Model
    model = SVC(decision_function_shape="ovo", C=50, class_weight=None, degree = 2,
                           gamma='scale', kernel = 'rbf', probability=True)
    
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)  # Probabilitats predites per al conjunt de test

    # Etiquetes de classe personalitzades
    class_labels = ["blues", "classical", "country", "disco", "hiphop", 
                    "jazz", "metal", "pop", "reggae", "rock"]

    # Generar la ROC Curve per a cada classe
    n_classes = len(np.unique(y_train))
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Gràfic de la ROC Curve
    plt.figure(figsize=(12, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve (SVM)", fontsize=16)
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
