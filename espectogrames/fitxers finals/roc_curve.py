import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd 


def processament(data, labels, img_size=(128,128)):
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        if os.path.isdir(genre_path):
            for img_file in os.listdir(genre_path):
                img_path = os.path.join(genre_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    features, _ = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                    data.append(features)
                    labels.append(genre)
            


def codificar_label(data):
    label_encoder = preprocessing.LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data, label_encoder

def definirXY_normalitzar(data):
    X = data.drop(['label'],axis=1)
    y = data['label'] #variable independent (a predir)
    columnes = X.columns
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X) #escalem 
    X = pd.DataFrame(np_scaled, columns=columnes)#nou dataset sense label i filename
    return X, y

if __name__ == "__main__":
    base_dir = "ACproject-14-grup/datasets/Data1/images_original"

    data, labels = [], []
    print("[INFO] Processant dades i extraient característiques d'audio...")
    processament(data, labels, img_size=(128,128))
    
    print("[INFO] Codificant etiquetes i normalitzant dades...")
    data = pd.DataFrame(data)
    data["label"] = labels

    data, label_encoder = codificar_label(data)
    X, y = definirXY_normalitzar(data)

    print("\n[INFO] Dividint dataset en conjunt train i test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111, stratify=y)

    print("\n[INFO] Crear i entrenar el model SVC amb hiperparàmetres especificats...")
    model = SVC(C=5, gamma="scale", kernel="rbf", probability=True, random_state=42)
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

